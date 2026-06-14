from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_listen_preview_manifest import SAFETY_FLAGS, iter_jsonl, write_json  # noqa: E402
from tools.run_no_reference_delivery_pipeline import run_pipeline  # noqa: E402


def safe_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:80] or "unknown"


def first_track_id(path: Path) -> str:
    try:
        for row in iter_jsonl(path):
            return str(row.get("track_id") or path.stem)
    except Exception:
        return path.stem
    return path.stem


def input_output_dir(output_root: Path, index: int, optimizer_result_jsonl: Path) -> Path:
    digest = hashlib.sha256(str(optimizer_result_jsonl).encode("utf-8")).hexdigest()[:10]
    return output_root / f"{index:04d}_{safe_slug(first_track_id(optimizer_result_jsonl))}_{digest}"


def expand_optimizer_inputs(paths: list[Path], output_root: Path) -> list[Path]:
    expanded: list[Path] = []
    split_root = output_root / "_split_optimizer_inputs"
    for file_index, path in enumerate(paths, start=1):
        rows = list(iter_jsonl(path))
        if len(rows) <= 1:
            expanded.append(path)
            continue
        for row_index, row in enumerate(rows, start=1):
            track_id = str(row.get("track_id") or path.stem)
            split_path = split_root / f"{file_index:04d}_{row_index:04d}_{safe_slug(track_id)}.jsonl"
            write_jsonl(split_path, [row])
            expanded.append(split_path)
    return expanded


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def batch_row_from_manifest(index: int, optimizer_result_jsonl: Path, output_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "optimizer_result_jsonl": str(optimizer_result_jsonl),
        "output_dir": str(output_dir),
        "track_id": first_track_id(optimizer_result_jsonl),
        "final_status": str(manifest.get("final_status") or "execution_failed"),
        "delivery_allowed": bool(manifest.get("delivery_allowed")),
        "presentation_status": str(manifest.get("presentation_status") or ""),
        "release_status": str(manifest.get("release_status") or ""),
        "accepted_delivery_candidate_rows": int(manifest.get("accepted_delivery_candidate_rows") or 0),
        "audio_copied": False,
        "audio_exported": False,
        "training_started": False,
        "models_changed": False,
        "cleanup_routes_added": False,
        "thresholds_relaxed": False,
        "pipeline_manifest": str(output_dir / "no_reference_delivery_pipeline_manifest.json"),
        "listen_preview_manifest": str(output_dir / "no_reference_listen_preview_manifest.json"),
    }


def execution_failed_row(index: int, optimizer_result_jsonl: Path, output_dir: Path, error: Exception) -> dict[str, Any]:
    return {
        "index": index,
        "optimizer_result_jsonl": str(optimizer_result_jsonl),
        "output_dir": str(output_dir),
        "track_id": first_track_id(optimizer_result_jsonl),
        "final_status": "execution_failed",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "accepted_delivery_candidate_rows": 0,
        "error_type": type(error).__name__,
        "error": str(error),
        "audio_copied": False,
        "audio_exported": False,
        "training_started": False,
        "models_changed": False,
        "cleanup_routes_added": False,
        "thresholds_relaxed": False,
        "pipeline_manifest": str(output_dir / "no_reference_delivery_pipeline_manifest.json"),
        "listen_preview_manifest": str(output_dir / "no_reference_listen_preview_manifest.json"),
    }


def build_batch_manifest(output_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("final_status") or "unknown") for row in rows)
    return {
        "manifest_type": "no_reference_delivery_batch",
        "output_root": str(output_root),
        "total_inputs": len(rows),
        "accepted_for_listen_pack": int(status_counts.get("accepted_for_listen_pack", 0)),
        "blocked_from_listen_pack": int(status_counts.get("blocked_from_listen_pack", 0)),
        "execution_failed": int(status_counts.get("execution_failed", 0)),
        "status_counts": dict(sorted(status_counts.items())),
        "audio_copied": False,
        "audio_exported": False,
        "can_mark_release_ready": False,
        "safety": dict(SAFETY_FLAGS),
    }


def write_summary(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# No-Reference Delivery Batch Summary",
        "",
        f"- total_inputs: {manifest['total_inputs']}",
        f"- accepted_for_listen_pack: {manifest['accepted_for_listen_pack']}",
        f"- blocked_from_listen_pack: {manifest['blocked_from_listen_pack']}",
        f"- execution_failed: {manifest['execution_failed']}",
        f"- audio_copied: {str(manifest['audio_copied']).lower()}",
        f"- audio_exported: {str(manifest['audio_exported']).lower()}",
        f"- can_mark_release_ready: {str(manifest['can_mark_release_ready']).lower()}",
        "",
        "## Rows",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['final_status']}: {row['track_id']} -> {row['output_dir']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_batch(
    *,
    optimizer_result_jsonls: list[Path],
    output_root: Path,
    window_seconds: float,
    window_step_seconds: float,
    max_windows_per_candidate: int,
    max_candidates: int,
    min_window_count: int,
    enable_phase_alignment_rescue: bool = False,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    expanded_inputs = expand_optimizer_inputs(optimizer_result_jsonls, output_root)
    for index, optimizer_result_jsonl in enumerate(expanded_inputs, start=1):
        item_output_dir = input_output_dir(output_root, index, optimizer_result_jsonl)
        try:
            pipeline_manifest = run_pipeline(
                optimizer_result_jsonl=optimizer_result_jsonl,
                output_dir=item_output_dir,
                window_seconds=window_seconds,
                window_step_seconds=window_step_seconds,
                max_windows_per_candidate=max_windows_per_candidate,
                max_candidates=max_candidates,
                min_window_count=min_window_count,
                enable_phase_alignment_rescue=enable_phase_alignment_rescue,
            )
            rows.append(batch_row_from_manifest(index, optimizer_result_jsonl, item_output_dir, pipeline_manifest))
        except Exception as exc:
            rows.append(execution_failed_row(index, optimizer_result_jsonl, item_output_dir, exc))

    rows_path = output_root / "no_reference_delivery_batch_rows.jsonl"
    manifest_path = output_root / "no_reference_delivery_batch_manifest.json"
    summary_path = output_root / "no_reference_delivery_batch_summary.md"
    write_jsonl(rows_path, rows)
    manifest = build_batch_manifest(output_root, rows)
    write_json(manifest_path, manifest)
    write_summary(summary_path, manifest, rows)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run dense-gated no-reference delivery pipeline over multiple optimizer outputs.")
    parser.add_argument("--optimizer-result-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=6.0)
    parser.add_argument("--window-step-seconds", type=float, default=6.0)
    parser.add_argument("--max-windows-per-candidate", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--min-window-count", type=int, default=5)
    parser.add_argument("--enable-phase-alignment-rescue", action="store_true")
    args = parser.parse_args(argv)

    manifest = run_batch(
        optimizer_result_jsonls=list(args.optimizer_result_jsonl),
        output_root=args.output_root,
        window_seconds=args.window_seconds,
        window_step_seconds=args.window_step_seconds,
        max_windows_per_candidate=args.max_windows_per_candidate,
        max_candidates=args.max_candidates,
        min_window_count=args.min_window_count,
        enable_phase_alignment_rescue=args.enable_phase_alignment_rescue,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
