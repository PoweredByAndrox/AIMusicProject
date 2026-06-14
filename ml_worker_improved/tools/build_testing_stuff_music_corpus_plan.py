from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402


AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac")
DEFAULT_TARGET_STEMS = ("vocal", "minus", "melody", "drums", "bass", "other")
DEFAULT_DOCKER_SOURCE_ROOT = "/host_testing_stuff_music"
DEFAULT_FORBIDDEN_ACTIONS = (
    "no_reference_required",
    "no_human_csv_required",
    "no_audio_copy",
    "no_audio_export",
    "no_audio_hash",
    "no_training_start",
    "no_fine_tune_start",
    "no_threshold_relaxation",
    "no_cleanup_debleed_denoise",
    "no_model_or_registry_change",
    "no_release_ready_claim",
)
SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "audio_read": False,
    "audio_copied": False,
    "audio_exported": False,
    "audio_hashed": False,
    "release_ready": False,
}


def stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")


def sanitize(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:96] or "track"


def audio_paths(source_root: Path, suffixes: Sequence[str]) -> list[Path]:
    allowed = {suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}" for suffix in suffixes}
    if not source_root.exists() or not source_root.is_dir():
        return []
    return sorted(
        (path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in allowed),
        key=lambda path: path.relative_to(source_root).as_posix().lower(),
    )


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def label_path(root_label: str, relative_path: str) -> str:
    if not root_label:
        return ""
    return f"{root_label.replace(chr(92), '/').rstrip('/')}/{relative_path}"


def normalize_path_label(path: Path | str) -> str:
    return str(path).replace("\\", "/").rstrip("/")


def require_docker_source_root(source_root: Path, required_root: str = DEFAULT_DOCKER_SOURCE_ROOT) -> None:
    required = normalize_path_label(required_root)
    if normalize_path_label(source_root) == required:
        return
    raise SystemExit(f"source_root_must_be_docker_mount:{required}")


def unique_track_id(base: str, seen: set[str], ordinal: int) -> str:
    if base not in seen:
        seen.add(base)
        return base
    candidate = f"{base}_{ordinal:04d}"
    while candidate in seen:
        ordinal += 1
        candidate = f"{base}_{ordinal:04d}"
    seen.add(candidate)
    return candidate


def build_row(
    *,
    source_root: Path,
    audio_path: Path,
    ordinal: int,
    seen_track_ids: set[str],
    docker_root_label: str,
    host_root_label: str,
    target_stems: Sequence[str],
) -> dict[str, Any]:
    relative_path = relative_posix(audio_path, source_root)
    track_id = unique_track_id(sanitize(audio_path.stem), seen_track_ids, ordinal)
    stat = audio_path.stat()
    return {
        "schema": "testing_stuff_music_corpus_plan_row.v1",
        "index": ordinal,
        "track_id": track_id,
        "relative_path": relative_path,
        "source_audio_path": str(audio_path),
        "docker_audio_path": label_path(docker_root_label, relative_path),
        "host_audio_path_hint": label_path(host_root_label, relative_path),
        "source_audio_suffix": audio_path.suffix.lower(),
        "source_audio_bytes": int(stat.st_size),
        "source_exists": True,
        "intended_use": "no_reference_evaluation_queue",
        "target_stems": list(target_stems),
        "candidate_generation_required": True,
        "evaluation_allowed": True,
        "reference_available": False,
        "reference_required": False,
        "training_allowed": False,
        "fine_tune_allowed": False,
        "can_start_training_now": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "next_route": "generate_mode_specific_candidates_then_no_reference_judge",
        "expected_acceleration_mode": "gpu_inference_for_candidate_generation_then_cpu_dsp_scoring",
        "forbidden_actions": list(DEFAULT_FORBIDDEN_ACTIONS),
    }


def count_by_suffix(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("source_audio_suffix") or "") for row in rows).most_common())


def build_summary(
    *,
    source_root: Path,
    rows: list[dict[str, Any]],
    total_audio_files: int,
    capped_audio_files: int,
    corpus_status: str,
    target_stems: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": "testing_stuff_music_corpus_summary.v1",
        "corpus_status": corpus_status,
        "source_root": str(source_root),
        "docker_runtime_required": True,
        "docker_source_root_required": DEFAULT_DOCKER_SOURCE_ROOT,
        "docker_runtime_validated": False,
        "docker_source_root_validated": False,
        "total_audio_files": total_audio_files,
        "evaluation_queue_count": len(rows),
        "capped_audio_files": capped_audio_files,
        "suffix_counts": count_by_suffix(rows),
        "target_stems": list(target_stems),
        "training_allowed_count": sum(1 for row in rows if row.get("training_allowed") is True),
        "fine_tune_allowed_count": sum(1 for row in rows if row.get("fine_tune_allowed") is True),
        "can_start_training_now_count": sum(1 for row in rows if row.get("can_start_training_now") is True),
        "reference_available_count": sum(1 for row in rows if row.get("reference_available") is True),
        "expected_acceleration_mode": "gpu_inference_for_candidate_generation_then_cpu_dsp_scoring",
        "audio_read": False,
        "audio_copied": False,
        "audio_exported": False,
        "audio_hashed": False,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "release_ready": False,
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(
    *,
    output_dir: Path | None,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "testing_stuff_music_corpus_manifest.v1",
        "manifest_type": "testing_stuff_music_corpus_plan",
        "corpus_status": summary["corpus_status"],
        "docker_runtime_required": True,
        "docker_source_root_required": DEFAULT_DOCKER_SOURCE_ROOT,
        "docker_runtime_validated": summary["docker_runtime_validated"],
        "docker_source_root_validated": summary["docker_source_root_validated"],
        "delivery_allowed": False,
        "presentation_status": "diagnostic_corpus_plan_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "training_allowed": False,
        "fine_tune_allowed": False,
        "can_start_training_now": False,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "cleanup_routes_added": False,
        "thresholds_relaxed": False,
        "audio_read": False,
        "audio_copied": False,
        "audio_exported": False,
        "audio_hashed": False,
        "output_files": {
            "plan_jsonl": str((output_dir or Path("")).joinpath("testing_stuff_music_corpus_plan.jsonl")) if output_dir else "",
            "manifest_json": str((output_dir or Path("")).joinpath("testing_stuff_music_corpus_manifest.json")) if output_dir else "",
            "summary_md": str((output_dir or Path("")).joinpath("testing_stuff_music_corpus_summary.md")) if output_dir else "",
        },
        "summary": summary,
        "safety": dict(SAFETY_FLAGS),
        "forbidden_actions": list(DEFAULT_FORBIDDEN_ACTIONS),
    }


def build_corpus_plan(
    *,
    source_root: Path,
    docker_root_label: str = "/host_testing_stuff_music",
    host_root_label: str = "",
    max_tracks: int = 0,
    suffixes: Sequence[str] = AUDIO_SUFFIXES,
    target_stems: Sequence[str] = DEFAULT_TARGET_STEMS,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    all_audio_paths = audio_paths(source_root, suffixes)
    corpus_status = "ready" if source_root.exists() and source_root.is_dir() else "blocked_missing_source_root"
    capped_audio_files = 0
    selected_audio_paths = list(all_audio_paths)
    if max_tracks > 0 and len(selected_audio_paths) > max_tracks:
        capped_audio_files = len(selected_audio_paths) - max_tracks
        selected_audio_paths = selected_audio_paths[:max_tracks]
    seen_track_ids: set[str] = set()
    rows = [
        build_row(
            source_root=source_root,
            audio_path=path,
            ordinal=index,
            seen_track_ids=seen_track_ids,
            docker_root_label=docker_root_label,
            host_root_label=host_root_label,
            target_stems=target_stems,
        )
        for index, path in enumerate(selected_audio_paths, start=1)
    ]
    if corpus_status == "ready" and not all_audio_paths:
        corpus_status = "blocked_no_audio_files"
    summary = build_summary(
        source_root=source_root,
        rows=rows,
        total_audio_files=len(all_audio_paths),
        capped_audio_files=capped_audio_files,
        corpus_status=corpus_status,
        target_stems=target_stems,
    )
    manifest = build_manifest(output_dir=output_dir, summary=summary)
    return {"rows": rows, "summary": summary, "manifest": manifest}


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Testing Stuff Music Corpus Plan",
        "",
        f"- corpus_status: {summary['corpus_status']}",
        f"- source_root: {summary['source_root']}",
        f"- total_audio_files: {summary['total_audio_files']}",
        f"- evaluation_queue_count: {summary['evaluation_queue_count']}",
        f"- capped_audio_files: {summary['capped_audio_files']}",
        f"- training_allowed_count: {summary['training_allowed_count']}",
        f"- fine_tune_allowed_count: {summary['fine_tune_allowed_count']}",
        f"- can_start_training_now_count: {summary['can_start_training_now_count']}",
        f"- reference_available_count: {summary['reference_available_count']}",
        f"- expected_acceleration_mode: {summary['expected_acceleration_mode']}",
        "",
        "## Suffix Counts",
        "",
    ]
    for suffix, count in summary["suffix_counts"].items():
        lines.append(f"- {suffix}: {count}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result["manifest"]["output_files"] = {
        "plan_jsonl": str(output_dir / "testing_stuff_music_corpus_plan.jsonl"),
        "manifest_json": str(output_dir / "testing_stuff_music_corpus_manifest.json"),
        "summary_md": str(output_dir / "testing_stuff_music_corpus_summary.md"),
    }
    write_jsonl(output_dir / "testing_stuff_music_corpus_plan.jsonl", result["rows"])
    write_json(output_dir / "testing_stuff_music_corpus_manifest.json", result["manifest"])
    write_summary_md(output_dir / "testing_stuff_music_corpus_summary.md", result["summary"])


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_testing_stuff_music_corpus_plan.py",
        entrypoint_path="/app/tools/build_testing_stuff_music_corpus_plan.py",
    )
    parser = argparse.ArgumentParser(description="Build a Docker-visible no-reference evaluation queue from testing_stuff_music.")
    parser.add_argument("--source-root", type=Path, default=Path(DEFAULT_DOCKER_SOURCE_ROOT))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--docker-root-label", default="/host_testing_stuff_music")
    parser.add_argument("--host-root-label", default="")
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--suffix", action="append", default=[])
    args = parser.parse_args(argv)
    require_docker_source_root(args.source_root)

    result = build_corpus_plan(
        source_root=args.source_root,
        docker_root_label=args.docker_root_label,
        host_root_label=args.host_root_label,
        max_tracks=args.max_tracks,
        suffixes=args.suffix or AUDIO_SUFFIXES,
        output_dir=args.output_dir,
    )
    result["summary"]["docker_runtime_validated"] = True
    result["summary"]["docker_source_root_validated"] = True
    result["manifest"]["docker_runtime_validated"] = True
    result["manifest"]["docker_source_root_validated"] = True
    write_outputs(result, args.output_dir)
    print(stable_json(result["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
