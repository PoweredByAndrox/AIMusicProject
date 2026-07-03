#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from pseudo_teacher_contract import PSEUDO_TEACHER_SCHEMA  # noqa: E402


MANIFEST_NAME = "pseudo_teacher_audio_training_manifest.json"
EXAMPLES_NAME = "pseudo_teacher_audio_training_examples.jsonl"
BLOCKED_NAME = "pseudo_teacher_audio_training_blocked_candidates.jsonl"
PROGRESS_NAME = "pseudo_teacher_audio_training_progress.jsonl"
SAFE_STEM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
START_TIME = time.monotonic()


def validate_stem_name(name: str) -> str:
    if not SAFE_STEM_NAME_PATTERN.match(str(name or "")):
        raise ValueError(f"unsafe stem name: {name!r}")
    return str(name)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        rows.append(payload)
    return rows


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
        "expected_acceleration_mode": "external_only_cpu_metadata_curation",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "metadata_only_pseudo_teacher_dataset_build",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:18]


def existing_file_reason(path_value: Any, reason_name: str) -> tuple[Path | None, list[str]]:
    raw = str(path_value or "").strip()
    if not raw:
        return None, [f"missing_{reason_name}"]
    path = Path(raw)
    if not path.exists() or not path.is_file():
        return path, [f"{reason_name}_not_found"]
    return path, []


def iter_music_stems(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("stems", "items", "entries"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item


def selected_path_for_stem(row: dict[str, Any], stem: str) -> str:
    explicit_key = f"selected_{stem}_path"
    if str(row.get(explicit_key) or "").strip():
        return str(row[explicit_key])
    if stem == "vocal" and str(row.get("selected_vocal_path") or "").strip():
        return str(row["selected_vocal_path"])
    return ""


def selected_candidate_id_for_stem(row: dict[str, Any], stem: str) -> str:
    explicit_key = f"selected_{stem}_candidate_id"
    if str(row.get(explicit_key) or "").strip():
        return str(row[explicit_key])
    if stem == "vocal" and str(row.get("selected_vocal_candidate_id") or "").strip():
        return str(row["selected_vocal_candidate_id"])
    return ""


def find_music_stem(
    stems: Sequence[dict[str, Any]],
    *,
    track_id: str,
    stem: str,
    target_path: str,
    selected_candidate_id: str = "",
) -> dict[str, Any] | None:
    target = str(target_path)
    candidate_id = str(selected_candidate_id or "").strip()
    for item in stems:
        if str(item.get("track_id") or "") != track_id:
            continue
        if str(item.get("stem_type") or "") != stem:
            continue
        item_paths = {str(item.get("audio_path") or ""), str(item.get("normalized_audio_path") or "")}
        if target and target in item_paths:
            return item
        item_candidate_ids = {
            str(item.get("candidate_id") or ""),
            str(item.get("stem_id") or ""),
            str(item.get("selected_candidate_id") or ""),
            str(item.get(f"selected_{stem}_candidate_id") or ""),
        }
        if candidate_id and candidate_id in item_candidate_ids:
            return item
    return None


def build_row(
    *,
    run_root: Path,
    run_manifest_json: Path,
    selected_candidates_jsonl: Path,
    music_intelligence_manifest_json: Path,
    selected_row: dict[str, Any],
    music_stem: dict[str, Any],
    source_audio_path: Path,
    target_audio_path: Path,
    stem: str,
) -> dict[str, Any]:
    track_id = str(selected_row.get("track_id") or music_stem.get("track_id") or "").strip()
    selected_candidate_id = selected_candidate_id_for_stem(selected_row, stem)
    row_id = stable_id([track_id, stem, str(source_audio_path), str(target_audio_path), selected_candidate_id])
    return {
        "schema": PSEUDO_TEACHER_SCHEMA,
        "id": row_id,
        "track_id": track_id,
        "stem_type": stem,
        "source_audio_path": str(source_audio_path),
        "target_audio_path": str(target_audio_path),
        "target_audio_sha256": sha256_file(target_audio_path),
        "split": "train",
        "trainable": True,
        "pseudo_teacher": True,
        "pseudo_teacher_source": "no_reference_runtime_selected_candidate",
        "pseudo_teacher_targets_are_ground_truth": False,
        "human_verified": False,
        "reference_verified": False,
        "permission_status": "pseudo_teacher_runtime_artifact",
        "release_ready": False,
        "delivery_allowed": False,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "source_run_root": str(run_root),
        "run_manifest_json": str(run_manifest_json),
        "selected_candidates_jsonl": str(selected_candidates_jsonl),
        "music_intelligence_manifest_json": str(music_intelligence_manifest_json),
        "selected_candidate_id": selected_candidate_id,
        "source_separator_config": str(music_stem.get("source_separator_config") or ""),
        "pair_compatibility_status": str(music_stem.get("pair_compatibility_status") or ""),
    }


def inspect_selected_candidate(
    *,
    run_root: Path,
    run_manifest_json: Path,
    selected_candidates_jsonl: Path,
    music_intelligence_manifest_json: Path,
    run_manifest: dict[str, Any],
    music_stems: Sequence[dict[str, Any]],
    selected_row: dict[str, Any],
    stem: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    track_id = str(selected_row.get("track_id") or "").strip()
    target_path_raw = selected_path_for_stem(selected_row, stem)
    selected_candidate_id = selected_candidate_id_for_stem(selected_row, stem)
    reasons: list[str] = []
    if not track_id:
        reasons.append("missing_track_id")
    if not target_path_raw:
        reasons.append("missing_selected_stem_path")
    if selected_row.get("release_ready") is True:
        reasons.append("selected_candidate_release_ready_true")
    if selected_row.get("can_publish_as_result") is True:
        reasons.append("selected_candidate_publishable")
    if selected_row.get("can_mark_release_ready") is True:
        reasons.append("selected_candidate_can_mark_release_ready")
    if run_manifest.get("release_ready") is True:
        reasons.append("run_manifest_release_ready_true")

    source_path, source_reasons = existing_file_reason(run_manifest.get("input_audio"), "source_audio")
    target_path, target_reasons = existing_file_reason(target_path_raw, "target_audio")
    reasons.extend(source_reasons)
    reasons.extend(target_reasons)

    music_stem = find_music_stem(
        music_stems,
        track_id=track_id,
        stem=stem,
        target_path=target_path_raw,
        selected_candidate_id=selected_candidate_id,
    )
    if music_stem is None:
        reasons.append("music_intelligence_stem_not_found")
    else:
        if music_stem.get("release_ready") is True:
            reasons.append("music_intelligence_release_ready_true")
        if music_stem.get("analysis_allowed") is not True:
            reasons.append("music_intelligence_analysis_not_allowed")
        if str(music_stem.get("quality_status") or "") != "accepted":
            reasons.append("music_intelligence_quality_not_accepted")
        blockers = music_stem.get("blockers") or []
        if isinstance(blockers, list) and blockers:
            reasons.append("music_intelligence_blockers_present")

    if reasons or source_path is None or target_path is None or music_stem is None:
        blocked = {
            "schema": "pseudo_teacher_audio_training_blocked_candidate.v1",
            "run_root": str(run_root),
            "track_id": track_id,
            "stem_type": stem,
            "selected_candidate_id": selected_candidate_id,
            "source_audio_path": str(run_manifest.get("input_audio") or ""),
            "target_audio_path": target_path_raw
            or str((music_stem or {}).get("normalized_audio_path") or (music_stem or {}).get("audio_path") or ""),
            "blocked_reasons": sorted(set(reasons)),
            "release_ready": False,
        }
        return None, blocked

    return (
        build_row(
            run_root=run_root,
            run_manifest_json=run_manifest_json,
            selected_candidates_jsonl=selected_candidates_jsonl,
            music_intelligence_manifest_json=music_intelligence_manifest_json,
            selected_row=selected_row,
            music_stem=music_stem,
            source_audio_path=source_path,
            target_audio_path=target_path,
            stem=stem,
        ),
        None,
    )


def blocked_music_stem_without_selection(
    *,
    run_root: Path,
    run_manifest: dict[str, Any],
    music_stem: dict[str, Any],
    stem: str,
) -> dict[str, Any]:
    target_audio_path = str(music_stem.get("normalized_audio_path") or music_stem.get("audio_path") or "")
    reasons = ["selected_candidates_empty"]
    source_path, source_reasons = existing_file_reason(run_manifest.get("input_audio"), "source_audio")
    target_path, target_reasons = existing_file_reason(target_audio_path, "target_audio")
    reasons.extend(source_reasons)
    reasons.extend(target_reasons)
    if music_stem.get("release_ready") is True:
        reasons.append("music_intelligence_release_ready_true")
    if music_stem.get("analysis_allowed") is not True:
        reasons.append("music_intelligence_analysis_not_allowed")
    if str(music_stem.get("quality_status") or "") != "accepted":
        reasons.append("music_intelligence_quality_not_accepted")
    blockers = music_stem.get("blockers") or []
    if isinstance(blockers, list) and blockers:
        reasons.append("music_intelligence_blockers_present")
    return {
        "schema": "pseudo_teacher_audio_training_blocked_candidate.v1",
        "run_root": str(run_root),
        "track_id": str(music_stem.get("track_id") or ""),
        "stem_type": stem,
        "selected_candidate_id": str(music_stem.get("candidate_id") or music_stem.get("stem_id") or ""),
        "source_audio_path": str(source_path or run_manifest.get("input_audio") or ""),
        "target_audio_path": str(target_path or target_audio_path),
        "blocked_reasons": sorted(set(reasons)),
        "release_ready": False,
    }


def build_pseudo_teacher_audio_training_examples(
    *,
    run_roots: Sequence[Path],
    output_dir: Path,
    stem: str = "vocal",
    scope: str = "pseudo_teacher_audio_training",
) -> dict[str, Any]:
    stem = validate_stem_name(stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", stem=stem, scope=scope, run_root_count=len(run_roots))

    rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for run_root in run_roots:
        run_manifest_json = run_root / "run_manifest.json"
        selected_candidates_jsonl = run_root / "selected_candidates.jsonl"
        music_intelligence_manifest_json = run_root / "music_intelligence_input_manifest.json"
        progress(progress_path, "run_root_start", stem=stem, run_root=str(run_root))
        if not run_manifest_json.exists() or not selected_candidates_jsonl.exists() or not music_intelligence_manifest_json.exists():
            blocked_rows.append(
                {
                    "schema": "pseudo_teacher_audio_training_blocked_candidate.v1",
                    "run_root": str(run_root),
                    "track_id": "",
                    "stem_type": stem,
                    "selected_candidate_id": "",
                    "source_audio_path": "",
                    "target_audio_path": "",
                    "blocked_reasons": [
                        reason
                        for reason, path in (
                            ("run_manifest_missing", run_manifest_json),
                            ("selected_candidates_missing", selected_candidates_jsonl),
                            ("music_intelligence_manifest_missing", music_intelligence_manifest_json),
                        )
                        if not path.exists()
                    ],
                    "release_ready": False,
                }
            )
            progress(progress_path, "run_root_finish", stem=stem, run_root=str(run_root), status="blocked_missing_manifest")
            continue
        run_manifest = read_json(run_manifest_json)
        selected_candidates = read_jsonl(selected_candidates_jsonl)
        music_manifest = read_json(music_intelligence_manifest_json)
        music_stems = list(iter_music_stems(music_manifest))
        if not selected_candidates:
            matching_music_stems = [item for item in music_stems if str(item.get("stem_type") or "") == stem]
            if matching_music_stems:
                for music_stem in matching_music_stems:
                    blocked_rows.append(
                        blocked_music_stem_without_selection(
                            run_root=run_root,
                            run_manifest=run_manifest,
                            music_stem=music_stem,
                            stem=stem,
                        )
                    )
            else:
                blocked_rows.append(
                    {
                        "schema": "pseudo_teacher_audio_training_blocked_candidate.v1",
                        "run_root": str(run_root),
                        "track_id": "",
                        "stem_type": stem,
                        "selected_candidate_id": "",
                        "source_audio_path": str(run_manifest.get("input_audio") or ""),
                        "target_audio_path": "",
                        "blocked_reasons": ["selected_candidates_empty", "music_intelligence_stem_not_found"],
                        "release_ready": False,
                    }
                )
            progress(
                progress_path,
                "run_root_finish",
                stem=stem,
                run_root=str(run_root),
                selected_candidate_count=0,
                matching_music_stem_count=len(matching_music_stems),
                cumulative_pseudo_teacher_example_count=len(rows),
                cumulative_blocked_candidate_count=len(blocked_rows),
            )
            continue
        for selected_row in selected_candidates:
            row, blocked = inspect_selected_candidate(
                run_root=run_root,
                run_manifest_json=run_manifest_json,
                selected_candidates_jsonl=selected_candidates_jsonl,
                music_intelligence_manifest_json=music_intelligence_manifest_json,
                run_manifest=run_manifest,
                music_stems=music_stems,
                selected_row=selected_row,
                stem=stem,
            )
            if blocked is not None:
                blocked_rows.append(blocked)
                continue
            if row is None:
                continue
            key = (row["track_id"], row["stem_type"], row["source_audio_path"], row["target_audio_path"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rows.append(row)
        progress(
            progress_path,
            "run_root_finish",
            stem=stem,
            run_root=str(run_root),
            selected_candidate_count=len(selected_candidates),
            cumulative_pseudo_teacher_example_count=len(rows),
            cumulative_blocked_candidate_count=len(blocked_rows),
        )

    examples_path = output_dir / EXAMPLES_NAME
    blocked_path = output_dir / BLOCKED_NAME
    write_jsonl(examples_path, rows)
    write_jsonl(blocked_path, blocked_rows)
    track_ids = sorted({row["track_id"] for row in rows})
    status = "passed" if rows else "failed"
    manifest = {
        "schema": "pseudo_teacher_audio_training_manifest.v1",
        "manifest_type": "pseudo_teacher_audio_training",
        "status": status,
        "scope": scope,
        "stem": stem,
        "output_dir": str(output_dir),
        "run_roots": [str(path) for path in run_roots],
        "pseudo_teacher_examples_jsonl": str(examples_path),
        "blocked_candidates_jsonl": str(blocked_path),
        "pseudo_teacher_example_count": len(rows),
        "blocked_candidate_count": len(blocked_rows),
        "track_ids": track_ids,
        "unique_track_count": len(track_ids),
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "pseudo_teacher_training_dataset_not_for_listen_pack",
        "release_status": "training_dataset_only_requires_verified_holdout",
        "release_ready": False,
        "progress_jsonl": str(progress_path),
        "safety": {
            "metadata_only": True,
            "pseudo_teacher_only": True,
            "dirty_stems_never_ground_truth": True,
            "requires_verified_holdout": True,
            "release_ready_forced_false": True,
            "no_threshold_relaxation": True,
        },
        "provenance": {
            "tool_path": str(Path(__file__).resolve()),
        },
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(
        progress_path,
        "finish",
        stem=stem,
        status=status,
        pseudo_teacher_example_count=len(rows),
        blocked_candidate_count=len(blocked_rows),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build explicit pseudo-teacher audio training rows from accepted runtime stems.")
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="vocal")
    parser.add_argument("--scope", default="pseudo_teacher_audio_training")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_pseudo_teacher_audio_training_examples.py",
        entrypoint_path="/app/tools/build_pseudo_teacher_audio_training_examples.py",
    )
    args = parse_args(argv)
    manifest = build_pseudo_teacher_audio_training_examples(
        run_roots=list(args.run_root),
        output_dir=args.output_dir,
        stem=str(args.stem),
        scope=str(args.scope),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "status": manifest["status"],
                "pseudo_teacher_example_count": manifest["pseudo_teacher_example_count"],
                "blocked_candidate_count": manifest["blocked_candidate_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
