from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402


SCHEMA = "ralph_residual_focus_windows.v1"
FOCUS_JSONL_NAME = "focus.jsonl"
MANIFEST_NAME = "manifest.json"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
SAFETY_FLAGS = {
    "delivery_allowed": False,
    "release_ready": False,
    "can_mark_release_ready": False,
    "can_publish_as_result": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "thresholds_relaxed": False,
    "audio_exported": False,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def manifest_track_id(manifest: dict[str, Any]) -> str:
    if manifest.get("track_id"):
        return str(manifest["track_id"])
    filters = manifest.get("focus_window_filters")
    if isinstance(filters, dict) and filters.get("track_id"):
        return str(filters["track_id"])
    raw_path = (
        manifest.get("inputs", {}).get("raw_audio_path")
        if isinstance(manifest.get("inputs"), dict)
        else ""
    )
    return Path(str(raw_path or "raw_audio")).stem


def selected_candidate_records(manifest: dict[str, Any], candidate_name: str = "") -> list[dict[str, Any]]:
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        return []
    selected = candidate_name or str(manifest.get("selected_candidate") or "")
    records = [row for row in candidates if isinstance(row, dict)]
    if selected:
        matched = [row for row in records if str(row.get("candidate_name") or "") == selected]
        if matched:
            return matched
    accepted = [
        row
        for row in records
        if isinstance(row.get("evaluation"), dict)
        and str(row["evaluation"].get("status") or "") == "accepted"
    ]
    return accepted[:1] if accepted else records[:1]


def rows_from_candidate(
    *,
    ralph_manifest: Path,
    manifest: dict[str, Any],
    candidate: dict[str, Any],
    track_id: str,
) -> list[dict[str, Any]]:
    candidate_name = str(candidate.get("candidate_name") or "")
    evaluation = candidate.get("evaluation") if isinstance(candidate.get("evaluation"), dict) else {}
    summary = evaluation.get("summary") if isinstance(evaluation.get("summary"), dict) else {}
    windows = summary.get("top_vocal_residue_windows")
    if not isinstance(windows, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            continue
        start = as_float(window.get("start_sec"))
        end = as_float(window.get("end_sec"))
        if end <= start:
            continue
        residue_corr = max(0.0, as_float(window.get("vocal_residue_corr")))
        block_probability = min(1.0, residue_corr)
        rows.append(
            {
                "schema": SCHEMA,
                "row_id": f"{track_id}__{candidate_name}__rfocus__{index:04d}",
                "track_id": track_id,
                "candidate_id": candidate_name,
                "route_id": "ralph_residual_focus_windows",
                "target_name": "minus_focus",
                "view_name": "ralph_top_vocal_residue_window",
                "critic_decision": "ralph_residual_focus",
                "true_label": 0,
                "predicted_accept_probability": round(1.0 - block_probability, 6),
                "predicted_block_probability": round(block_probability, 6),
                "predicted_blockers": ["ralph_top_vocal_residue_window"],
                "window_start_sec": round(start, 6),
                "window_end_sec": round(end, 6),
                "vocal_residue_corr": residue_corr,
                "vocal_projection_ratio_db": as_float(window.get("vocal_projection_ratio_db")),
                "source_manifest": str(ralph_manifest),
                "source_manifest_mode": str(manifest.get("mode") or ""),
                "source_candidate_status": str(evaluation.get("status") or ""),
                "separator_training_started": False,
                "shadow_only": True,
                "release_ready": False,
            }
        )
    return rows


def summary_text(manifest: dict[str, Any]) -> str:
    summary = manifest["summary"]
    return "\n".join(
        [
            "# RALPH Residual Focus Windows",
            "",
            f"- focus_window_count: `{summary['focus_window_count']}`",
            f"- selected_candidate: `{summary['selected_candidate']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            f"- output_status: `{manifest['output_status']}`",
            "",
        ]
    )


def build_ralph_residual_focus_windows(
    *,
    ralph_manifest: Path,
    output_dir: Path,
    max_windows: int = 8,
    candidate_name: str = "",
    command_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    append_jsonl(progress_path, {"event": "start", "ralph_manifest": str(ralph_manifest)})

    manifest = read_json(ralph_manifest)
    track_id = manifest_track_id(manifest)
    candidates = selected_candidate_records(manifest, candidate_name)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rows.extend(
            rows_from_candidate(
                ralph_manifest=ralph_manifest,
                manifest=manifest,
                candidate=candidate,
                track_id=track_id,
            )
        )
    rows.sort(
        key=lambda row: (
            -as_float(row.get("vocal_residue_corr")),
            as_float(row.get("window_start_sec")),
            str(row.get("candidate_id") or ""),
        )
    )
    if max_windows > 0:
        rows = rows[:max_windows]

    focus_jsonl = output_dir / FOCUS_JSONL_NAME
    write_jsonl(focus_jsonl, rows)
    selected_candidate = str(candidates[0].get("candidate_name") or "") if candidates else ""
    result = {
        "schema": SCHEMA,
        "mode": "ralph_residual_focus_windows",
        "ralph_manifest": str(ralph_manifest),
        "ralph_manifest_sha256": sha256_file(ralph_manifest),
        "output_dir": str(output_dir),
        "focus_windows_jsonl": str(focus_jsonl),
        "manifest_json": str(output_dir / MANIFEST_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "output_status": "focus_windows_ready" if rows else "blocked_no_residual_windows",
        "presentation_status": "diagnostic_focus_windows_not_for_listen_pack",
        "release_status": "not_release_ready",
        "summary": {
            "track_id": track_id,
            "selected_candidate": selected_candidate,
            "candidate_count": len(candidates),
            "focus_window_count": len(rows),
            "max_windows": int(max_windows),
        },
        "command_args": list(command_args or []),
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_NAME, result)
    (output_dir / SUMMARY_NAME).write_text(summary_text(result), encoding="utf-8")
    append_jsonl(
        progress_path,
        {
            "event": "finish",
            "output_status": result["output_status"],
            "focus_window_count": len(rows),
            "focus_windows_jsonl": str(focus_jsonl),
        },
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build focus rows from RALPH top vocal-residue windows.")
    parser.add_argument("--ralph-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--candidate-name", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_ralph_residual_focus_windows.py",
        entrypoint_path="/app/tools/build_ralph_residual_focus_windows.py",
    )
    args = parse_args(argv)
    manifest = build_ralph_residual_focus_windows(
        ralph_manifest=args.ralph_manifest,
        output_dir=args.output_dir,
        max_windows=int(args.max_windows),
        candidate_name=str(args.candidate_name or ""),
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["output_status"] == "focus_windows_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
