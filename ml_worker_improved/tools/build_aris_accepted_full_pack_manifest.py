from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from artifact_name_policy import validate_artifact_relative_path
from docker_runtime import require_docker_runtime


MANIFEST_NAME = "manifest.json"
SOURCE_NAME = "source.json"
PROGRESS_NAME = "progress.jsonl"
SUMMARY_NAME = "summary.md"
PAIR_STEMS = ("vocal", "minus")
SIXS_STEMS = ("drums", "bass", "other", "guitar", "piano")
START_TIME = time.monotonic()


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


def _ensure_path_inside_rag(*, artifact_rag_root: Path, path: Path, label: str) -> Path:
    root = _resolved(artifact_rag_root)
    target = _resolved(path)
    if target != root and root not in target.parents:
        raise ValueError(f"{label} must be inside artifact_rag_root")
    return target


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _append_progress(path: Path, event: str, **fields: Any) -> None:
    row = {
        "event": event,
        "phase": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_manifest_build",
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "source_manifest_build_only",
        "delivery_allowed": False,
        "release_ready": False,
    }
    row.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _first_accepted(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if (
            row.get("delivery_status") == "accepted"
            and row.get("delivery_allowed") is True
            and row.get("presentation_status") == "accepted_for_listen_pack"
        ):
            return row
    raise ValueError("accepted_jsonl does not contain an accepted delivery row")


def _candidate_by_stem(rows: list[dict[str, Any]], *, track_alias: str) -> dict[str, dict[str, Any]]:
    by_stem: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("track_alias") or "") != track_alias:
            continue
        stem = str(row.get("stem") or "")
        if stem in PAIR_STEMS and stem not in by_stem:
            by_stem[stem] = row
    missing = [stem for stem in PAIR_STEMS if stem not in by_stem]
    if missing:
        raise ValueError(f"accepted candidates missing required stems: {', '.join(missing)}")
    return by_stem


def _audio_stem_payload(
    path_value: str,
    *,
    artifact_rag_root: Path,
    source: str,
    source_manifest_json: Path,
    stem: str,
) -> dict[str, Any]:
    path = _resolved(Path(path_value))
    if path != artifact_rag_root and artifact_rag_root not in path.parents:
        raise ValueError(f"stem audio path must be inside artifact_rag_root: {stem}")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"stem audio missing: {path}")
    return {
        "path": str(path),
        "source": source,
        "source_manifest_json": str(source_manifest_json),
    }


def _summary_text(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ARIS Accepted Full Pack Source Manifest",
            "",
            f"- status: `{manifest['status']}`",
            f"- stem_count: `{manifest['stem_count']}`",
            "- delivery_allowed: `false`",
            "- audio_copied: `false`",
            "",
        ]
    )


def build_aris_accepted_full_pack_manifest(
    *,
    artifact_rag_root: Path,
    accepted_jsonl: Path,
    candidates_jsonl: Path,
    sixs_stem_manifest_json: Path,
    output_dir: Path,
    command_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    root, output = _ensure_output_inside_rag(artifact_rag_root=artifact_rag_root, output_dir=output_dir)
    accepted_path = _ensure_path_inside_rag(artifact_rag_root=root, path=accepted_jsonl, label="accepted_jsonl")
    candidates_path = _ensure_path_inside_rag(artifact_rag_root=root, path=candidates_jsonl, label="candidates_jsonl")
    sixs_path = _ensure_path_inside_rag(
        artifact_rag_root=root,
        path=sixs_stem_manifest_json,
        label="sixs_stem_manifest_json",
    )
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / PROGRESS_NAME
    progress_path.write_text("", encoding="utf-8")
    _append_progress(
        progress_path,
        "run_start",
        accepted_jsonl=str(accepted_path),
        candidates_jsonl=str(candidates_path),
        sixs_stem_manifest_json=str(sixs_path),
    )
    accepted_rows = list(_iter_jsonl(accepted_path))
    candidates = list(_iter_jsonl(candidates_path))
    accepted = _first_accepted(accepted_rows)
    track_alias = str(accepted.get("track_alias") or "")
    pair_candidates = _candidate_by_stem(candidates, track_alias=track_alias)
    sixs_manifest = _read_json(sixs_path)
    sixs_stems = sixs_manifest.get("stems") if isinstance(sixs_manifest.get("stems"), dict) else {}
    stems: dict[str, dict[str, Any]] = {}
    for stem in PAIR_STEMS:
        stems[stem] = _audio_stem_payload(
            str(pair_candidates[stem].get("target_audio_path") or ""),
            artifact_rag_root=root,
            source="accepted_vocal_minus_matrix",
            source_manifest_json=candidates_path,
            stem=stem,
        )
    for stem in SIXS_STEMS:
        payload = sixs_stems.get(stem)
        if not isinstance(payload, dict) or not payload.get("path"):
            raise ValueError(f"6s stem missing from source manifest: {stem}")
        stems[stem] = _audio_stem_payload(
            str(payload["path"]),
            artifact_rag_root=root,
            source="htdemucs_6s",
            source_manifest_json=sixs_path,
            stem=stem,
        )
    source_manifest = {
        "schema": "aris_accepted_full_pack_source_manifest.v1",
        "audio_path": str(sixs_manifest.get("audio_path") or ""),
        "model_name": "accepted_vocal_minus_plus_htdemucs_6s",
        "track_alias": track_alias,
        "track_id": str(pair_candidates["vocal"].get("track_id") or track_alias),
        "objective_evidence": dict(accepted.get("objective_evidence") or {}),
        "source_acceptance_row": accepted,
        "stems": stems,
        "contract": {
            "registered_model_candidate": True,
            "fallback_used": False,
            "hidden_inference": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "accepted_vocal_minus_matrix_required": True,
            "sixs_stem_manifest_required": True,
        },
        "execution_contract": dict(sixs_manifest.get("execution_contract") or {}),
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_ready": False,
        "audio_exported": False,
        "models_changed": False,
        "thresholds_relaxed": False,
    }
    source_path = output / SOURCE_NAME
    _write_json(source_path, source_manifest)
    manifest = {
        "schema": "aris_accepted_full_pack_source_manifest_manifest.v1",
        "manifest_type": "aris_accepted_full_pack_source_manifest",
        "status": "ready_for_pack_build",
        "artifact_rag_root": str(root),
        "output_dir": str(output),
        "accepted_jsonl": str(accepted_path),
        "candidates_jsonl": str(candidates_path),
        "sixs_stem_manifest_json": str(sixs_path),
        "source_manifest_json": str(source_path),
        "summary_md": str(output / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "track_alias": track_alias,
        "stem_count": len(stems),
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_ready": False,
        "audio_exported": False,
        "models_changed": False,
        "thresholds_relaxed": False,
        "command_args": list(command_args or []),
    }
    _write_json(output / MANIFEST_NAME, manifest)
    (output / SUMMARY_NAME).write_text(_summary_text(manifest), encoding="utf-8")
    _append_progress(progress_path, "run_finish", status=manifest["status"], manifest_json=str(output / MANIFEST_NAME))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build accepted vocal/minus + 6s source manifest for full pack.")
    parser.add_argument("--artifact-rag-root", type=Path, required=True)
    parser.add_argument("--accepted-jsonl", type=Path, required=True)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--sixs-stem-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_aris_accepted_full_pack_manifest.py",
        entrypoint_path="/app/tools/build_aris_accepted_full_pack_manifest.py",
    )
    args = parse_args(argv)
    manifest = build_aris_accepted_full_pack_manifest(
        artifact_rag_root=args.artifact_rag_root,
        accepted_jsonl=args.accepted_jsonl,
        candidates_jsonl=args.candidates_jsonl,
        sixs_stem_manifest_json=args.sixs_stem_manifest_json,
        output_dir=args.output_dir,
        command_args=list(argv or sys.argv[1:]),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
