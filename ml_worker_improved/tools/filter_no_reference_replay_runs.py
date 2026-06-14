from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_replay_bundle import expanded_runtime_run_roots  # noqa: E402
from tools.render_no_reference_audio_preview import PREVIEW_STATUSES  # noqa: E402

BLOCKED_COMPLETE_STATUSES = {"preview_render_skipped", "preview_render_failed", "preview_blocked_safety_contract"}
PREVIEW_BOOLEAN_FIELDS = ("accepted_pair_ready", "manifest_preview_ready", "audio_preview_ready")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def iter_json_values(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_json_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_json_values(child)


def true_flag_found(run_dir: Path, keys: set[str]) -> bool:
    for path in sorted(run_dir.rglob("*.json")) + sorted(run_dir.rglob("*.jsonl")):
        if not path.is_file():
            continue
        try:
            rows: list[Any]
            if path.suffix == ".jsonl":
                rows = read_jsonl(path)
            else:
                rows = [read_json(path)]
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            for value in iter_json_values(row):
                if isinstance(value, dict) and any(value.get(key) is True for key in keys):
                    return True
    return False


def rendered_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("render_status") == "rendered" and row.get("validation_status") in {"validated", "accepted", "ok"})


def explicit_failure_or_skip_reason(preview: dict[str, Any], status: str) -> str:
    if status == "preview_render_skipped":
        return str(preview.get("skip_reason") or "")
    return str(preview.get("failure_reason") or preview.get("preview_failure_reason") or "")


def classify_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_manifest_path = run_dir / "run_manifest.json"
    preview_path = run_dir / "preview_manifest.json"
    contract_path = run_dir / "preview_render_contract.json"
    rows_path = run_dir / "rendered_preview_audio.jsonl"
    run_manifest = read_json(run_manifest_path)
    preview = read_json(preview_path)
    contract = read_json(contract_path)
    rows = read_jsonl(rows_path)
    reasons: list[str] = []

    if not run_manifest_path.exists():
        reasons.append("run_manifest_missing")
    if not preview_path.exists():
        reasons.append("preview_manifest_missing")
    if not contract_path.exists():
        reasons.append("preview_render_contract_missing")
    if true_flag_found(run_dir, {"release_ready", "can_mark_release_ready", "can_publish_as_result"}):
        reasons.append("release_ready_true_found")

    preview_status = str(preview.get("preview_status") or preview.get("runtime_status") or "")
    if preview_status == "preview_ready":
        reasons.append("ambiguous_legacy_preview_ready")
    if preview_status not in PREVIEW_STATUSES and preview_status != "preview_ready":
        reasons.append("unknown_preview_status")
    if preview_status not in BLOCKED_COMPLETE_STATUSES and not rows_path.exists():
        reasons.append("rendered_preview_audio_jsonl_missing")
    if preview_status in BLOCKED_COMPLETE_STATUSES and not explicit_failure_or_skip_reason(preview, preview_status):
        reasons.append("preview_skip_or_failure_reason_missing")
    if preview_status not in BLOCKED_COMPLETE_STATUSES and rows_path.exists() and not isinstance(rows, list):
        reasons.append("rendered_audio_count_undetermined")
    if preview.get("audio_preview_ready") is True and rendered_count(rows) <= 0:
        reasons.append("audio_ready_without_rendered_audio_rows")
    if preview_status == "audio_preview_ready" and rendered_count(rows) <= 0:
        reasons.append("audio_preview_ready_requires_rendered_audio")
    missing_preview_fields = [field for field in PREVIEW_BOOLEAN_FIELDS if field not in preview]
    if missing_preview_fields and preview_status not in BLOCKED_COMPLETE_STATUSES:
        reasons.append("preview_1_fields_missing:" + ",".join(missing_preview_fields))
    if contract.get("release_ready") is True:
        reasons.append("contract_release_ready_true")

    audio_ready = preview_status == "audio_preview_ready" and rendered_count(rows) > 0 and not reasons
    return {
        "run_root": str(run_dir),
        "preview_contract_complete": not reasons,
        "preview_status": preview_status,
        "audio_preview_ready": bool(audio_ready),
        "rendered_preview_audio_count": rendered_count(rows),
        "incomplete_reasons": sorted(set(reasons)),
        "release_ready": False,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
    }


def filter_replay_runs(*, run_roots: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    expanded = expanded_runtime_run_roots(run_roots)
    rows = [classify_run(run_dir) for run_dir in expanded]
    complete_rows = [row for row in rows if row["preview_contract_complete"] is True]
    incomplete_rows = [row for row in rows if row["preview_contract_complete"] is not True]
    write_jsonl(output_dir / "preview_contract_complete_runs.jsonl", complete_rows)
    write_jsonl(output_dir / "preview_contract_incomplete_runs.jsonl", incomplete_rows)
    summary = {
        "total_runs_scanned": len(rows),
        "preview_contract_complete_runs": len(complete_rows),
        "preview_contract_incomplete_runs": len(incomplete_rows),
        "audio_preview_ready_count": sum(1 for row in complete_rows if row.get("audio_preview_ready") is True),
        "rendered_preview_audio_count": sum(int(row.get("rendered_preview_audio_count") or 0) for row in complete_rows),
        "release_ready_count": 0,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
        "release_ready": False,
    }
    write_json(output_dir / "preview_contract_filter_summary.json", summary)
    lines = [
        "# Preview-Contract Replay Run Filter",
        "",
        f"- total_runs_scanned: {summary['total_runs_scanned']}",
        f"- preview_contract_complete_runs: {summary['preview_contract_complete_runs']}",
        f"- preview_contract_incomplete_runs: {summary['preview_contract_incomplete_runs']}",
        f"- audio_preview_ready_count: {summary['audio_preview_ready_count']}",
        f"- rendered_preview_audio_count: {summary['rendered_preview_audio_count']}",
        "- release_ready: false",
        "- melody_transcription_ran: false",
        "- recommendation_index_updated: false",
    ]
    (output_dir / "preview_contract_filter_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter no-reference replay runs to PREVIEW-1 contract-complete runs.")
    parser.add_argument("--run-roots", nargs="*", type=Path, default=[])
    parser.add_argument("--parent-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = list(args.run_roots)
    if args.parent_dir is not None:
        roots.append(args.parent_dir)
    if not roots:
        raise SystemExit("--run-roots or --parent-dir is required")
    summary = filter_replay_runs(run_roots=roots, output_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
