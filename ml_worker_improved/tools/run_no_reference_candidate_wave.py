from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import soundfile as sf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_ml_dataset import stable_hash, write_jsonl  # noqa: E402


ALLOWED_PROFILES = {"safe", "balanced", "exploration"}
ALLOWED_TARGET_MODES = {"vocal", "minus", "both"}
DETERMINISTIC_CREATED_AT = "1970-01-01T00:00:00Z"

SAFETY_FLAGS = {
    "codex_required_at_runtime": False,
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "separator_training_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "release_ready": False,
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def sanitize(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    return cleaned.strip("._-")[:96] or "candidate"


def stable_id(prefix: str, payload: dict[str, Any]) -> str:
    return sanitize(f"{prefix}_{stable_hash(payload)[:16]}")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_input_audio(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "reason": "input_audio_missing", "path": str(path)}
    if path.stat().st_size <= 0:
        return {"ok": False, "reason": "input_audio_empty", "path": str(path)}
    try:
        info = sf.info(str(path))
    except Exception as exc:  # pragma: no cover - defensive corrupt input path.
        return {"ok": False, "reason": f"input_audio_unreadable:{type(exc).__name__}", "path": str(path)}
    return {
        "ok": True,
        "path": str(path),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": round(float(info.frames) / float(info.samplerate), 6) if info.samplerate else 0.0,
    }


def default_source_paths(output_dir: Path) -> list[Path]:
    return [
        output_dir / "candidate_wave_sources.jsonl",
        output_dir / "mock_separator_outputs.jsonl",
        output_dir / "separator_outputs.jsonl",
    ]


def load_source_rows(output_dir: Path, source_jsonl: Path | None) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = [source_jsonl] if source_jsonl is not None else default_source_paths(output_dir)
    existing = [path for path in paths if path.exists()]
    rows: list[dict[str, Any]] = []
    for path in existing:
        for row in iter_jsonl(path):
            rows.append({**row, "source_jsonl": str(path)})
    rows.sort(key=lambda row: (str(row.get("candidate_config_id") or ""), str(row.get("target_mode") or ""), str(row.get("vocal_path") or ""), str(row.get("minus_path") or row.get("instrumental_path") or "")))
    return rows, existing


def source_hash(row: dict[str, Any]) -> str:
    stable_payload = {
        key: row.get(key)
        for key in sorted(row)
        if key not in {"created_at", "source_jsonl"}
    }
    return stable_hash(stable_payload)


def target_mode(row: dict[str, Any]) -> str:
    text = str(row.get("target_mode") or row.get("mode") or "").strip().lower()
    if text in {"vocals"}:
        return "vocal"
    if text in {"instrumental"}:
        return "minus"
    return text


def audio_path(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def failure_row(
    *,
    run_id: str,
    track_id: str,
    input_audio: Path,
    row: dict[str, Any],
    skip_reason: str,
    profile: str,
) -> dict[str, Any]:
    mode = target_mode(row)
    config_id = str(row.get("candidate_config_id") or row.get("config_id") or "unknown_config")
    family = str(row.get("candidate_family") or row.get("family") or mode or "unknown_family")
    return {
        "run_id": run_id,
        "track_id": track_id,
        "input_audio_path": str(input_audio),
        "mix_path": str(input_audio),
        "candidate_id": stable_id("failed", {"run_id": run_id, "config": config_id, "mode": mode, "reason": skip_reason}),
        "candidate_family": family,
        "candidate_config_id": config_id,
        "target_mode": mode or "unknown",
        "vocal_path": audio_path(row, "vocal_path"),
        "minus_path": audio_path(row, "minus_path", "instrumental_path"),
        "instrumental_path": audio_path(row, "minus_path", "instrumental_path"),
        "can_use_for_vocal": False,
        "can_use_for_minus": False,
        "can_use_for_pair": False,
        "generation_profile": profile,
        "generation_status": "skipped",
        "skip_reason": skip_reason,
        "source_tool": str(row.get("source_tool") or "existing_local_separator_output"),
        "source_model_id": str(row.get("source_model_id") or ""),
        "source_config_hash": source_hash(row),
        "created_at": DETERMINISTIC_CREATED_AT,
        "audio_rendered": False,
        "release_ready": False,
        **SAFETY_FLAGS,
    }


def copy_stem(source: Path, output_dir: Path, candidate_id: str, stem: str) -> str:
    suffix = source.suffix if source.suffix else ".wav"
    target = output_dir / candidate_id / f"{stem}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        raise ValueError(f"candidate_output_would_overwrite_source:{source}")
    shutil.copy2(source, target)
    return str(target)


def materialize_asset(
    *,
    run_id: str,
    track_id: str,
    input_audio: Path,
    output_dir: Path,
    row: dict[str, Any],
    profile: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    mode = target_mode(row)
    if mode not in ALLOWED_TARGET_MODES:
        return None, failure_row(run_id=run_id, track_id=track_id, input_audio=input_audio, row=row, skip_reason="unsupported_target_mode", profile=profile)
    vocal_source = audio_path(row, "vocal_path")
    minus_source = audio_path(row, "minus_path", "instrumental_path")
    missing: list[str] = []
    if mode in {"vocal", "both"} and (not vocal_source or not Path(vocal_source).exists()):
        missing.append("vocal_path")
    if mode in {"minus", "both"} and (not minus_source or not Path(minus_source).exists()):
        missing.append("minus_path")
    if missing:
        return None, failure_row(run_id=run_id, track_id=track_id, input_audio=input_audio, row=row, skip_reason="missing_" + "_and_".join(missing), profile=profile)
    config_id = str(row.get("candidate_config_id") or row.get("config_id") or stable_hash(row)[:12])
    family = str(row.get("candidate_family") or row.get("family") or f"{mode}_focused")
    candidate_id = stable_id(
        mode,
        {
            "input_audio": str(input_audio),
            "profile": profile,
            "candidate_config_id": config_id,
            "target_mode": mode,
            "source_config_hash": source_hash(row),
        },
    )
    try:
        copied_vocal = copy_stem(Path(vocal_source), output_dir, candidate_id, "vocals") if vocal_source and mode in {"vocal", "both"} else ""
        copied_minus = copy_stem(Path(minus_source), output_dir, candidate_id, "minus") if minus_source and mode in {"minus", "both"} else ""
    except Exception as exc:
        return None, failure_row(run_id=run_id, track_id=track_id, input_audio=input_audio, row=row, skip_reason=f"copy_failed:{type(exc).__name__}", profile=profile)
    return {
        "run_id": run_id,
        "track_id": track_id,
        "input_audio_path": str(input_audio),
        "mix_path": str(input_audio),
        "candidate_id": candidate_id,
        "candidate_family": family,
        "candidate_config_id": config_id,
        "target_mode": mode,
        "vocal_path": copied_vocal,
        "minus_path": copied_minus,
        "instrumental_path": copied_minus,
        "can_use_for_vocal": bool(copied_vocal),
        "can_use_for_minus": bool(copied_minus),
        "can_use_for_pair": bool(copied_vocal and copied_minus),
        "generation_profile": profile,
        "generation_status": "generated",
        "skip_reason": "",
        "source_tool": str(row.get("source_tool") or "existing_local_separator_output"),
        "source_model_id": str(row.get("source_model_id") or ""),
        "source_config_hash": source_hash(row),
        "created_at": DETERMINISTIC_CREATED_AT,
        "audio_rendered": False,
        "release_ready": False,
        **SAFETY_FLAGS,
    }, None


def pair_candidate(vocal: dict[str, Any], minus: dict[str, Any], *, max_index: int) -> dict[str, Any]:
    candidate_id = stable_id(
        "pair",
        {
            "vocal_candidate_id": vocal["candidate_id"],
            "minus_candidate_id": minus["candidate_id"],
            "vocal_config": vocal["candidate_config_id"],
            "minus_config": minus["candidate_config_id"],
            "index": max_index,
        },
    )
    return {
        "run_id": vocal["run_id"],
        "track_id": vocal["track_id"],
        "input_audio_path": vocal["input_audio_path"],
        "mix_path": vocal["mix_path"],
        "candidate_id": candidate_id,
        "candidate_family": f"{vocal['candidate_family']}+{minus['candidate_family']}",
        "candidate_config_id": f"{vocal['candidate_config_id']}+{minus['candidate_config_id']}",
        "target_mode": "both",
        "vocal_path": vocal["vocal_path"],
        "minus_path": minus["minus_path"],
        "instrumental_path": minus["minus_path"],
        "can_use_for_vocal": True,
        "can_use_for_minus": True,
        "can_use_for_pair": True,
        "generation_profile": vocal["generation_profile"],
        "generation_status": "generated",
        "skip_reason": "",
        "source_tool": "+".join(sorted({vocal["source_tool"], minus["source_tool"]})),
        "source_model_id": "+".join(sorted({value for value in (vocal["source_model_id"], minus["source_model_id"]) if value})),
        "source_config_hash": stable_hash({"vocal": vocal["source_config_hash"], "minus": minus["source_config_hash"]}),
        "created_at": DETERMINISTIC_CREATED_AT,
        "audio_rendered": False,
        "release_ready": False,
        "is_complete_pair": True,
        **SAFETY_FLAGS,
    }


def build_pairs(assets: list[dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    direct_pairs = [row for row in assets if row["can_use_for_pair"] is True]
    vocals = [row for row in assets if row["can_use_for_vocal"] is True]
    minuses = [row for row in assets if row["can_use_for_minus"] is True]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in sorted(direct_pairs, key=lambda item: (item["candidate_config_id"], item["candidate_id"])):
        key = (row["vocal_path"], row["minus_path"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({**row, "is_complete_pair": True})
    pair_index = 0
    for vocal in sorted(vocals, key=lambda item: (item["candidate_config_id"], item["candidate_id"])):
        for minus in sorted(minuses, key=lambda item: (item["candidate_config_id"], item["candidate_id"])):
            key = (vocal["vocal_path"], minus["minus_path"])
            if key in seen:
                continue
            seen.add(key)
            pair_index += 1
            rows.append(pair_candidate(vocal, minus, max_index=pair_index))
    rows.sort(key=lambda row: (row["candidate_config_id"], row["candidate_id"], row["vocal_path"], row["minus_path"]))
    limit = max(0, int(max_candidates))
    return rows[:limit] if limit else rows


def build_summary(
    *,
    source_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    source_paths: list[Path],
    input_validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_wave_executed": True,
        "source_jsonl_paths": [str(path) for path in source_paths],
        "input_validation": input_validation,
        "candidates_attempted": len(source_rows),
        "candidates_generated": len(rows),
        "complete_pairs": sum(1 for row in rows if row.get("is_complete_pair") is True),
        "vocal_usable_candidates": sum(1 for row in rows if row.get("can_use_for_vocal") is True),
        "minus_usable_candidates": sum(1 for row in rows if row.get("can_use_for_minus") is True),
        "pair_usable_candidates": sum(1 for row in rows if row.get("can_use_for_pair") is True),
        "failed_candidates": len(failures),
        "skip_reasons": dict(sorted({reason: sum(1 for row in failures if row.get("skip_reason") == reason) for reason in {str(row.get("skip_reason") or "") for row in failures}}.items())),
        "release_ready_count": sum(1 for row in rows + failures if row.get("release_ready") is True),
        **SAFETY_FLAGS,
    }


def run_candidate_wave(
    *,
    input_audio: Path,
    output_dir: Path,
    candidate_pool_jsonl: Path,
    profile: str = "safe",
    source_jsonl: Path | None = None,
    max_candidates: int = 6,
) -> dict[str, Any]:
    if profile not in ALLOWED_PROFILES:
        raise ValueError(f"Unsupported candidate profile: {profile}")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_validation = validate_input_audio(input_audio)
    source_rows, source_paths = load_source_rows(output_dir, source_jsonl)
    track_id = sanitize(input_audio.stem)
    run_id = stable_id(
        "candidate_wave",
        {
            "input_audio_path": str(input_audio),
            "input_audio_sha256": file_digest(input_audio) if input_validation["ok"] else "",
            "profile": profile,
            "source_rows": [{key: row.get(key) for key in sorted(row)} for row in source_rows],
        },
    )
    assets: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not input_validation["ok"]:
        failures.append(
            failure_row(
                run_id=run_id,
                track_id=track_id,
                input_audio=input_audio,
                row={"candidate_config_id": "input_audio", "candidate_family": "input_validation", "target_mode": "both"},
                skip_reason=str(input_validation["reason"]),
                profile=profile,
            )
        )
    else:
        for row in source_rows:
            asset, failure = materialize_asset(run_id=run_id, track_id=str(row.get("track_id") or track_id), input_audio=input_audio, output_dir=output_dir, row=row, profile=profile)
            if asset is not None:
                assets.append(asset)
            if failure is not None:
                failures.append(failure)
    rows = build_pairs(assets, max_candidates=max_candidates)
    summary = build_summary(source_rows=source_rows, rows=rows, failures=failures, source_paths=source_paths, input_validation=input_validation)
    write_jsonl(candidate_pool_jsonl, rows)
    write_jsonl(output_dir / "candidate_wave_failures.jsonl", failures)
    write_json(output_dir / "candidate_wave_summary.json", summary)
    write_json(
        output_dir / "candidate_wave_manifest.json",
        {
            "run_id": run_id,
            "input_audio_path": str(input_audio),
            "candidate_pool_jsonl": str(candidate_pool_jsonl),
            "profile": profile,
            "max_candidates": int(max_candidates),
            "summary": summary,
            **SAFETY_FLAGS,
        },
    )
    return {"run_id": run_id, "rows": rows, "failures": failures, "summary": summary}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an autonomous no-reference candidate generation/collection wave.")
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool-jsonl", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(sorted(ALLOWED_PROFILES)), default="safe")
    parser.add_argument("--source-jsonl", type=Path)
    parser.add_argument("--max-candidates", type=int, default=6)
    args = parser.parse_args(argv)
    result = run_candidate_wave(
        input_audio=args.input_audio,
        output_dir=args.output_dir,
        candidate_pool_jsonl=args.candidate_pool_jsonl,
        profile=args.profile,
        source_jsonl=args.source_jsonl,
        max_candidates=args.max_candidates,
    )
    print(json.dumps({"candidate_pool_jsonl": str(args.candidate_pool_jsonl), **result["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
