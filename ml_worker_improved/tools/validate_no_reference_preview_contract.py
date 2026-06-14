from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import soundfile as sf


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.render_no_reference_audio_preview import PREVIEW_STATUSES, iter_jsonl, read_json  # noqa: E402


def is_inside(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_rendered_row(run_dir: Path, row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if row.get("release_ready") is True:
        blockers.append("release_ready_forbidden")
    if row.get("render_status") != "rendered":
        return blockers
    rendered_path = Path(str(row.get("rendered_path") or ""))
    if not rendered_path.exists():
        blockers.append("rendered_audio_missing")
        return blockers
    if not is_inside(run_dir, rendered_path):
        blockers.append("rendered_audio_outside_run_dir")
    try:
        info = sf.info(str(rendered_path))
    except Exception as exc:  # pragma: no cover - defensive corrupt artifact path.
        blockers.append(f"rendered_audio_unreadable:{type(exc).__name__}")
        return blockers
    if int(info.frames) <= 0:
        blockers.append("rendered_audio_empty")
    if int(info.samplerate) != int(row.get("sample_rate") or 0):
        blockers.append("rendered_audio_sample_rate_mismatch")
    if int(info.channels) != int(row.get("channels") or 0):
        blockers.append("rendered_audio_channels_mismatch")
    if row.get("validation_status") != "validated":
        blockers.append("rendered_audio_not_validated")
    return blockers


def validate_preview_contract(*, run_dir: Path) -> dict[str, Any]:
    preview_path = run_dir / "preview_manifest.json"
    rows_path = run_dir / "rendered_preview_audio.jsonl"
    contract_path = run_dir / "preview_render_contract.json"
    blockers: list[str] = []
    if not preview_path.exists():
        blockers.append("preview_manifest_missing")
    if not rows_path.exists():
        blockers.append("rendered_preview_audio_jsonl_missing")
    if not contract_path.exists():
        blockers.append("preview_render_contract_missing")
    preview = read_json(preview_path)
    rows = iter_jsonl(rows_path)
    preview_status = str(preview.get("preview_status") or preview.get("runtime_status") or "")
    if preview_status not in PREVIEW_STATUSES and preview_status != "preview_ready":
        blockers.append("unknown_preview_status")
    if preview_status == "preview_ready" and not preview.get("rendered_preview_audio"):
        blockers.append("ambiguous_preview_ready_without_audio")
    if preview.get("release_ready") is True:
        blockers.append("release_ready_forbidden")
    if preview.get("can_publish_as_result") is True:
        blockers.append("can_publish_as_result_forbidden")
    if preview.get("can_mark_release_ready") is True:
        blockers.append("can_mark_release_ready_forbidden")
    rendered_rows = [row for row in rows if row.get("render_status") == "rendered"]
    rendered_paths = [str(row.get("rendered_path") or "") for row in rendered_rows if row.get("rendered_path")]
    if preview.get("audio_preview_ready") is True and not rendered_rows:
        blockers.append("audio_ready_without_rendered_audio_rows")
    if preview_status == "audio_preview_ready" and len(rendered_rows) < 2:
        blockers.append("audio_preview_ready_requires_vocal_and_minus")
    for row in rows:
        blockers.extend(validate_rendered_row(run_dir, row))
    manifest_paths = [str(path) for path in preview.get("rendered_preview_audio") or [] if path]
    missing_manifest_paths = sorted(set(manifest_paths) - set(rendered_paths))
    if missing_manifest_paths:
        blockers.append("preview_manifest_paths_missing_from_rendered_rows")
    return {
        "ok": not blockers,
        "run_dir": str(run_dir),
        "preview_status": preview_status,
        "rendered_preview_audio_count": len(rendered_rows),
        "blockers": sorted(set(blockers)),
        "release_ready": False,
        "can_publish_as_result": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate no-reference audio preview render contract artifacts.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_preview_contract(run_dir=args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
