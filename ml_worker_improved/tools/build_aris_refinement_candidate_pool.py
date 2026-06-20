#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from artifact_name_policy import track_alias, validate_artifact_relative_path, validate_artifact_segment  # noqa: E402
from docker_runtime import require_docker_runtime  # noqa: E402
from run_aris_refinement_matrix import FORBIDDEN_PRIMARY_DIRECTIONS  # noqa: E402

CANDIDATE_POOL_NAME = "candidate_pool.jsonl"
SKIPPED_NAME = "skipped.jsonl"
MANIFEST_NAME = "manifest.json"
SUMMARY_NAME = "summary.md"
GATE_KEYS = ("reference_gate", "no_reference_gate", "pair_consistency_gate")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _ensure_output_inside_rag(*, artifact_rag_root: Path, output_dir: Path) -> tuple[Path, Path]:
    root = _resolved(artifact_rag_root)
    output = _resolved(output_dir)
    if output != root and root not in output.parents:
        raise ValueError("output_dir must be inside artifact_rag_root")
    try:
        validate_artifact_relative_path(output.relative_to(root).as_posix())
    except ValueError as exc:
        raise ValueError("output_dir must use short Windows-safe relative path inside artifact_rag_root") from exc
    return root, output


def _ensure_path_inside_rag(
    *,
    artifact_rag_root: Path,
    path: Path | str,
    label: str,
    base_dir: Path | None = None,
) -> Path:
    if not str(path or "").strip():
        raise ValueError(f"{label} is required")
    candidate = Path(path)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    root = _resolved(artifact_rag_root)
    resolved = _resolved(candidate)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must be inside artifact_rag_root")
    return resolved


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _truthy_contract_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _normalize_token(value) in {"1", "true", "yes", "y", "used"}


def _falsey_contract_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    return _normalize_token(value) in {"0", "false", "no", "n", "unavailable", "missing", "pruned"}


def _provenance_flag(key: str, *sources: dict[str, Any]) -> bool:
    return any(_truthy_contract_flag(source.get(key)) for source in sources if isinstance(source, dict))


def _audio_bytes_available(*sources: dict[str, Any]) -> bool:
    if _provenance_flag("metadata_only", *sources):
        return False
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("binary_audio_available", "audio_bytes_available", "binary_audio_model_bytes_available"):
            if key in source and _falsey_contract_flag(source.get(key)):
                return False
    return True


def _validate_family(value: Any) -> str:
    family = _normalize_token(value)
    if family in FORBIDDEN_PRIMARY_DIRECTIONS:
        raise ValueError(f"forbidden primary direction: {family}")
    return family


def _next_unused_track_alias(aliases: dict[str, str]) -> str:
    used = set(aliases.values())
    index = len(aliases) + 1
    while track_alias(index) in used:
        index += 1
    return track_alias(index)


def _stable_track_alias(track_id: str, aliases: dict[str, str], *, requested_alias: str = "") -> str:
    if track_id not in aliases:
        if requested_alias and requested_alias not in set(aliases.values()):
            aliases[track_id] = requested_alias
        else:
            aliases[track_id] = _next_unused_track_alias(aliases)
    return aliases[track_id]


def _candidate_alias(family: str, stem: str, index: int) -> str:
    prefix = "v" if stem == "vocal" else "m" if stem == "minus" else "c"
    return f"{prefix}{index:03d}"


def _short_audio_path(raw_path: str, *, track_alias_value: str, stem: str) -> str:
    raw = str(raw_path or "").replace("\\", "/")
    if raw.startswith("audio/"):
        validate_artifact_relative_path(raw)
        return raw
    short_path = f"audio/{track_alias_value}/{stem}.flac"
    validate_artifact_relative_path(short_path)
    return short_path


def _skip_row(job: dict[str, Any], *, reason: str, wave_manifest_json: Path) -> dict[str, Any]:
    return {
        "schema": "aris_refinement_candidate_skip_row.v1",
        "track_id": str(job.get("track_id") or ""),
        "stem": _normalize_token(job.get("stem_type") or job.get("stem")),
        "family": _normalize_token(job.get("source_candidate_family") or job.get("family")),
        "candidate_alias": str(job.get("id") or ""),
        "execution_manifest_path": str(job.get("execution_manifest_path") or ""),
        "wave_manifest_json": str(wave_manifest_json),
        "skip_reason": reason,
        "delivery_allowed": False,
        "release_ready": False,
        "fallback_used": _truthy_contract_flag(job.get("fallback_used")),
        "fallback_as_success": _truthy_contract_flag(job.get("fallback_as_success")),
    }


def _row_from_job(
    job: dict[str, Any],
    *,
    execution_manifest: dict[str, Any],
    wave_manifest_json: Path,
    track_aliases: dict[str, str],
    row_index: int,
) -> dict[str, Any]:
    family = _validate_family(job.get("source_candidate_family") or job.get("family"))
    track_id = str(job.get("track_id") or job.get("id") or f"track-{row_index}")
    stem = _normalize_token(job.get("stem_type") or job.get("stem") or execution_manifest.get("inputs", {}).get("target_stem"))
    if stem == "instrumental":
        stem = "minus"
    if stem not in {"vocal", "minus"}:
        raise ValueError(f"unsupported stem for ARIS refinement candidate pool: {stem}")
    requested_track_alias = str(job.get("track_alias") or "")
    if requested_track_alias:
        validate_artifact_segment(requested_track_alias)
    track_alias_value = _stable_track_alias(track_id, track_aliases, requested_alias=requested_track_alias)
    validate_artifact_segment(track_alias_value)
    candidate_manifest = execution_manifest.get("candidate_manifest") if isinstance(execution_manifest.get("candidate_manifest"), dict) else {}
    fallback_used = _provenance_flag("fallback_used", job, execution_manifest, candidate_manifest)
    fallback_as_success = _provenance_flag("fallback_as_success", job, execution_manifest, candidate_manifest)
    audio_available = _audio_bytes_available(job, execution_manifest, candidate_manifest)
    target_audio_path = str(candidate_manifest.get("target_audio_path") or "")
    short_audio_path = _short_audio_path(target_audio_path, track_alias_value=track_alias_value, stem=stem)
    status = _normalize_token(job.get("candidate_status") or "")
    if not status:
        status = "generated" if _normalize_token(job.get("status")) == "completed" and _normalize_token(execution_manifest.get("status")) == "completed" else "failed"
    candidate_alias = str(job.get("candidate_alias") or _candidate_alias(family, stem, row_index))
    validate_artifact_segment(candidate_alias)
    row = {
        "schema": "aris_refinement_candidate_row.v1",
        "track_alias": track_alias_value,
        "track_id": track_id,
        "stem": stem,
        "family": family,
        "model_id": str(job.get("source_candidate_model_id") or execution_manifest.get("model_id") or ""),
        "candidate_alias": candidate_alias,
        "candidate_status": status,
        "short_audio_path": short_audio_path,
        "source_audio_path": str(job.get("source_audio_path") or ""),
        "target_audio_path": target_audio_path,
        "candidate_manifest_path": str(candidate_manifest.get("path") or ""),
        "execution_manifest_path": str(job.get("execution_manifest_path") or ""),
        "wave_manifest_json": str(wave_manifest_json),
        "delivery_allowed": False,
        "release_ready": False,
        "fallback_used": fallback_used,
        "fallback_as_success": fallback_as_success,
        "binary_audio_available": audio_available,
        "audio_bytes_available": audio_available,
        "metadata_only": not audio_available,
    }
    failed_checks = job.get("failed_checks") or job.get("failed_check_names")
    if failed_checks:
        row["failed_checks"] = failed_checks
    for key in GATE_KEYS:
        if job.get(key) is not None:
            row[key] = str(job[key])
        elif execution_manifest.get(key) is not None:
            row[key] = str(execution_manifest[key])
    return row


def build_aris_refinement_candidate_pool(
    *,
    artifact_rag_root: Path,
    wave_manifest_jsons: Sequence[Path],
    output_dir: Path,
    command_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    root, output = _ensure_output_inside_rag(artifact_rag_root=artifact_rag_root, output_dir=output_dir)
    if not wave_manifest_jsons:
        raise ValueError("at least one --wave-manifest-json is required")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    wave_manifest_paths: list[Path] = []
    track_aliases: dict[str, str] = {}
    for wave_manifest_json in wave_manifest_jsons:
        wave_manifest_path = _ensure_path_inside_rag(
            artifact_rag_root=root,
            path=wave_manifest_json,
            label="wave_manifest_json",
        )
        wave_manifest_paths.append(wave_manifest_path)
        wave_manifest = read_json(wave_manifest_path)
        jobs_jsonl = _ensure_path_inside_rag(
            artifact_rag_root=root,
            path=str(wave_manifest.get("jobs_jsonl") or ""),
            label="jobs_jsonl",
            base_dir=wave_manifest_path.parent,
        )
        if not jobs_jsonl.exists():
            raise FileNotFoundError(f"wave jobs_jsonl missing: {jobs_jsonl}")
        for job in iter_jsonl(jobs_jsonl):
            _validate_family(job.get("source_candidate_family") or job.get("family"))
            execution_manifest_path = _ensure_path_inside_rag(
                artifact_rag_root=root,
                path=str(job.get("execution_manifest_path") or ""),
                label="execution_manifest_path",
                base_dir=jobs_jsonl.parent,
            )
            if not execution_manifest_path.exists():
                skipped.append(_skip_row(job, reason="execution_manifest_missing", wave_manifest_json=wave_manifest_path))
                continue
            execution_manifest = read_json(execution_manifest_path)
            if _normalize_token(execution_manifest.get("status")) != "completed":
                skipped.append(_skip_row(job, reason="execution_manifest_not_completed", wave_manifest_json=wave_manifest_path))
                continue
            rows.append(
                _row_from_job(
                    job,
                    execution_manifest=execution_manifest,
                    wave_manifest_json=wave_manifest_path,
                    track_aliases=track_aliases,
                    row_index=len(rows) + 1,
                )
            )
    candidate_pool_jsonl = output / CANDIDATE_POOL_NAME
    skipped_jsonl = output / SKIPPED_NAME
    summary_md = output / SUMMARY_NAME
    manifest_json = output / MANIFEST_NAME
    write_jsonl(candidate_pool_jsonl, rows)
    write_jsonl(skipped_jsonl, skipped)
    manifest = {
        "schema": "aris_refinement_candidate_pool_manifest.v1",
        "manifest_type": "aris_refinement_candidate_pool",
        "artifact_rag_root": str(root),
        "output_dir": str(output),
        "wave_manifest_jsons": [str(path) for path in wave_manifest_paths],
        "candidate_pool_jsonl": str(candidate_pool_jsonl),
        "skipped_jsonl": str(skipped_jsonl),
        "summary_md": str(summary_md),
        "candidate_count": len(rows),
        "skipped_count": len(skipped),
        "track_count": len({row["track_alias"] for row in rows}),
        "delivery_allowed": False,
        "release_ready": False,
        "fallback_used": False,
        "safety": {
            "audio_read": False,
            "audio_exported": False,
            "models_changed": False,
            "thresholds_relaxed": False,
            "objective_gate_evidence_invented": False,
        },
        "command_args": list(command_args or []),
    }
    write_json(manifest_json, manifest)
    summary_md.write_text(
        "\n".join(
            [
                "# ARIS Refinement Candidate Pool",
                "",
                f"- candidate_count: {len(rows)}",
                f"- skipped_count: {len(skipped)}",
                f"- track_count: {manifest['track_count']}",
                "- delivery_allowed: false",
                "- release_ready: false",
                "- objective_gate_evidence_invented: false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAG-controlled ARIS refinement candidate pool from registered wave manifests.")
    parser.add_argument("--artifact-rag-root", type=Path, required=True)
    parser.add_argument("--wave-manifest-json", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_aris_refinement_candidate_pool.py",
        entrypoint_path="/app/tools/build_aris_refinement_candidate_pool.py",
    )
    args = parse_args(argv)
    manifest = build_aris_refinement_candidate_pool(
        artifact_rag_root=args.artifact_rag_root,
        wave_manifest_jsons=args.wave_manifest_json,
        output_dir=args.output_dir,
        command_args=list(argv or sys.argv[1:]),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
