from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.render_no_reference_audio_preview import render_audio_preview_contract  # noqa: E402


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def status_for_manifest(manifest: dict[str, Any]) -> str:
    preview_status = str(manifest.get("preview_status") or "")
    if preview_status == "audio_preview_ready":
        return "backfill_succeeded"
    if preview_status == "preview_blocked_safety_contract":
        return "backfill_blocked_safety_contract"
    if preview_status == "preview_render_skipped":
        return "backfill_skipped"
    return "backfill_failed"


def annotate_backfill(run_dir: Path, *, selected_before: str, manifest: dict[str, Any]) -> dict[str, Any]:
    selected_after = read_text(run_dir / "selected_candidates.jsonl")
    backfill_status = status_for_manifest(manifest)
    selected_modified = selected_before != selected_after
    annotations = {
        "backfill_attempted": True,
        "backfill_status": backfill_status,
        "separator_jobs_rerun": False,
        "selected_candidates_modified": selected_modified,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "release_ready": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
    }
    manifest = {**manifest, **annotations}
    write_json(run_dir / "preview_manifest.json", manifest)
    contract_path = run_dir / "preview_render_contract.json"
    contract = read_json(contract_path)
    contract.update(
        {
            **annotations,
            "accepted_pair_ready": manifest.get("accepted_pair_ready") is True,
            "manifest_preview_ready": manifest.get("manifest_preview_ready") is True,
            "audio_preview_ready": manifest.get("audio_preview_ready") is True,
            "preview_render_attempted": manifest.get("preview_render_attempted") is True,
            "preview_render_skipped": manifest.get("preview_render_skipped") is True,
            "preview_render_failed": manifest.get("preview_render_failed") is True,
            "preview_status": str(manifest.get("preview_status") or ""),
            "validation_status": str(manifest.get("validation_status") or ""),
            "failure_reason": str(manifest.get("failure_reason") or ""),
            "skip_reason": str(manifest.get("skip_reason") or ""),
            "rendered_preview_audio_count": int(manifest.get("rendered_preview_audio_count") or 0),
        }
    )
    write_json(contract_path, contract)
    with (run_dir / "preview_render_summary.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"- backfill_status: {backfill_status}\n")
        handle.write("- separator_jobs_rerun: false\n")
        handle.write(f"- selected_candidates_modified: {str(selected_modified).lower()}\n")
    return manifest


def backfill_preview_contract(*, run_dir: Path, preview_duration_sec: float = 30.0, preview_render_mode: str = "bounded") -> dict[str, Any]:
    run_dir = Path(run_dir)
    selected_before = read_text(run_dir / "selected_candidates.jsonl")
    manifest = render_audio_preview_contract(
        run_dir=run_dir,
        preview_duration_sec=float(preview_duration_sec),
        preview_render_mode=preview_render_mode,
        enable_audio_preview_render=True,
        require_preview_audio_for_preview_ready=True,
    )
    manifest = annotate_backfill(run_dir, selected_before=selected_before, manifest=manifest)
    return {
        "run_dir": str(run_dir),
        "preview_status": str(manifest.get("preview_status") or ""),
        "audio_preview_ready": manifest.get("audio_preview_ready") is True,
        "rendered_preview_audio_count": int(manifest.get("rendered_preview_audio_count") or 0),
        "backfill_attempted": True,
        "backfill_status": str(manifest.get("backfill_status") or ""),
        "separator_jobs_rerun": False,
        "selected_candidates_modified": manifest.get("selected_candidates_modified") is True,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "release_ready": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill PREVIEW-1 preview contract artifacts from existing selected stems.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preview-duration-sec", type=float, default=30.0)
    parser.add_argument("--preview-render-mode", choices=("bounded", "full"), default="bounded")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = backfill_preview_contract(run_dir=args.run_dir, preview_duration_sec=args.preview_duration_sec, preview_render_mode=args.preview_render_mode)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
