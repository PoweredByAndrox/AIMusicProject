from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_status_contract import sha256_file  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402


SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "audio_rendered": False,
    "audio_exported": False,
    "audio_copied": False,
    "release_ready": False,
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path) if path.exists() else ""}


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_source_stem(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"vocal", "vocals"}:
        return "vocal"
    if text in {"minus", "instrumental"}:
        return "minus"
    if text == "both":
        return "both"
    return "unknown"


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str, float, float]:
    return (
        str(row.get("track_id") or ""),
        str(row.get("base_candidate_id") or ""),
        str(row.get("alternate_candidate_id") or ""),
        normalize_source_stem(row.get("source_stem")),
        round(as_float(row.get("window_start_sec")), 6),
        round(as_float(row.get("window_duration_sec")), 6),
    )


def preview_paths(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("rendered_vocal_preview_path") or row.get("preview_vocal_path") or ""),
        str(row.get("rendered_minus_preview_path") or row.get("preview_minus_path") or ""),
    )


def judge_paths(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("preview_vocal_path") or row.get("rendered_vocal_preview_path") or ""),
        str(row.get("preview_minus_path") or row.get("rendered_minus_preview_path") or ""),
    )


def source_mix_path(
    preview_row: dict[str, Any],
    judge_row: dict[str, Any],
    contract_row: dict[str, Any],
) -> str:
    for row in (judge_row, preview_row, contract_row):
        for key in ("mix_path", "source_audio_path"):
            value = str(row.get(key) or "")
            if value:
                return value
    return ""


def first_present_path(*rows_and_keys: tuple[dict[str, Any], str]) -> str:
    for row, key in rows_and_keys:
        value = str(row.get(key) or "")
        if value:
            return value
    return ""


def index_by_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str, str, float, float], list[dict[str, Any]]]:
    indexed: dict[tuple[str, str, str, str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        indexed.setdefault(row_key(row), []).append(row)
    return indexed


def matching_judge_row(
    preview_row: dict[str, Any],
    judge_index: dict[tuple[str, str, str, str, float, float], list[dict[str, Any]]],
) -> dict[str, Any] | None:
    expected_paths = preview_paths(preview_row)
    for row in judge_index.get(row_key(preview_row), []):
        if judge_paths(row) == expected_paths:
            return row
    return None


def matching_contract_row(
    preview_row: dict[str, Any],
    contract_index: dict[tuple[str, str, str, str, float, float], list[dict[str, Any]]],
) -> dict[str, Any] | None:
    for row in contract_index.get(row_key(preview_row), []):
        if str(row.get("render_status") or "") == "render_ready":
            return row
    return None


def source_stem_type(rows: list[dict[str, Any]]) -> str:
    stems = {normalize_source_stem(row.get("source_stem")) for row in rows}
    stems.discard("unknown")
    if len(stems) == 1:
        return next(iter(stems))
    return "both"


def skip_reason(
    preview_row: dict[str, Any],
    *,
    judge_row: dict[str, Any] | None,
    contract_row: dict[str, Any] | None,
) -> str:
    vocal_path, minus_path = preview_paths(preview_row)
    if not vocal_path or not minus_path or not Path(vocal_path).exists() or not Path(minus_path).exists():
        return "missing_preview_audio"
    if str(preview_row.get("render_status") or "") != "render_ready":
        return "preview_not_render_ready"
    if preview_row.get("requires_post_render_judge") is not True:
        return "missing_post_render_judge_requirement"
    if not all(preview_row.get(key) is True for key in ("sample_rate_preserved", "channels_preserved", "duration_preserved")):
        return "preview_shape_not_preserved"
    if preview_row.get("can_publish_as_result") is True or preview_row.get("can_mark_release_ready") is True:
        return "preview_claims_publish_or_release"
    if contract_row is None:
        return "render_contract_missing"
    if contract_row.get("can_publish_as_result") is True or contract_row.get("can_mark_release_ready") is True:
        return "render_contract_claims_publish_or_release"
    if judge_row is None:
        return "preview_judge_missing"
    if str(judge_row.get("preview_decision") or "") != "preview_improved_window":
        return "preview_judge_not_improved"
    if str(judge_row.get("allowed_next_step") or "") != "expand_preview_policy":
        return "preview_judge_next_step_not_expand"
    if as_float(judge_row.get("score_delta")) <= 0.0:
        return "preview_score_delta_not_positive"
    if int(as_float(judge_row.get("blocker_delta"))) > 0:
        return "preview_added_blocker"
    if judge_row.get("can_publish_as_result") is True or judge_row.get("can_mark_release_ready") is True:
        return "preview_judge_claims_publish_or_release"
    return ""


def accepted_row(preview_row: dict[str, Any], judge_row: dict[str, Any], contract_row: dict[str, Any]) -> dict[str, Any]:
    vocal_path, minus_path = preview_paths(preview_row)
    mix_path = source_mix_path(preview_row, judge_row, contract_row)
    return {
        "track_id": str(preview_row.get("track_id") or ""),
        "base_candidate_id": str(preview_row.get("base_candidate_id") or ""),
        "alternate_candidate_id": str(preview_row.get("alternate_candidate_id") or ""),
        "source_stem": normalize_source_stem(preview_row.get("source_stem")),
        "mix_path": mix_path,
        "source_audio_path": mix_path,
        "window_start_sec": round(as_float(preview_row.get("window_start_sec")), 6),
        "window_duration_sec": round(as_float(preview_row.get("window_duration_sec")), 6),
        "selected_vocal_path": vocal_path,
        "selected_minus_path": minus_path,
        "preview_vocal_path": vocal_path,
        "preview_minus_path": minus_path,
        "base_window_score": as_float(judge_row.get("base_window_score")),
        "preview_window_score": as_float(judge_row.get("preview_window_score")),
        "score_delta": as_float(judge_row.get("score_delta")),
        "preview_decision": str(judge_row.get("preview_decision") or ""),
        "allowed_next_step": str(judge_row.get("allowed_next_step") or ""),
        "base_vocal_path": first_present_path((judge_row, "base_vocal_path"), (contract_row, "base_vocal_path")),
        "base_minus_path": first_present_path((judge_row, "base_minus_path"), (contract_row, "base_minus_path")),
        "alt_vocal_path": first_present_path((judge_row, "alt_vocal_path"), (contract_row, "alt_vocal_path")),
        "alt_minus_path": first_present_path((judge_row, "alt_minus_path"), (contract_row, "alt_minus_path")),
        "render_contract": {
            "render_status": str(contract_row.get("render_status") or ""),
            "can_render_preview_later": contract_row.get("can_render_preview_later") is True,
        },
    }


def selected_audio_path(row: dict[str, Any]) -> str:
    source_stem = normalize_source_stem(row.get("source_stem"))
    if source_stem == "minus":
        return str(row.get("selected_minus_path") or "")
    return str(row.get("selected_vocal_path") or "")


def accepted_manifest(
    *,
    preview_manifest_json: Path,
    preview_window_judge_jsonl: Path,
    render_contract_jsonl: Path,
    rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = rows[0]
    stem_type = source_stem_type(rows)
    selected_vocal = str(primary.get("selected_vocal_path") or "")
    selected_minus = str(primary.get("selected_minus_path") or "")
    mix_path = str(primary.get("mix_path") or primary.get("source_audio_path") or "")
    return {
        "manifest_type": "no_reference_segment_preview_listen",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": stem_type,
        "track_id": str(primary.get("track_id") or ""),
        "candidate_id": f"{primary.get('base_candidate_id')}__{primary.get('alternate_candidate_id')}__segment_preview",
        "mix_path": mix_path,
        "source_audio_path": mix_path,
        "selected_audio_path": selected_audio_path(primary),
        "selected_vocal_path": selected_vocal,
        "selected_minus_path": selected_minus,
        "selected_artifacts": {
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "listen_selected_vocals_flac": selected_vocal,
            "listen_minus_flac": selected_minus,
            "listen_segment_preview_vocal_flac": selected_vocal,
            "listen_segment_preview_minus_flac": selected_minus,
        },
        "accepted_count": len(rows),
        "skipped_count": len(skipped_rows),
        "accepted_listen_preview_rows": rows,
        "skipped_rows": skipped_rows,
        "source_preview_manifest": source_ref(preview_manifest_json),
        "source_preview_window_judge_jsonl": source_ref(preview_window_judge_jsonl),
        "source_render_contract_jsonl": source_ref(render_contract_jsonl),
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "release_ready": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": dict(SAFETY_FLAGS),
    }


def blocked_manifest(
    *,
    preview_manifest_json: Path,
    preview_window_judge_jsonl: Path,
    render_contract_jsonl: Path,
    skipped_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons = Counter(str(row.get("skip_reason") or "") for row in skipped_rows)
    return {
        "manifest_type": "no_reference_segment_preview_listen",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "no_post_render_improved_preview_rows",
        "stem_type": "unknown",
        "accepted_count": 0,
        "skipped_count": len(skipped_rows),
        "accepted_listen_preview_rows": [],
        "skipped_rows": skipped_rows,
        "skip_reason_counts": dict(sorted(reasons.items())),
        "source_preview_manifest": source_ref(preview_manifest_json),
        "source_preview_window_judge_jsonl": source_ref(preview_window_judge_jsonl),
        "source_render_contract_jsonl": source_ref(render_contract_jsonl),
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "release_ready": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(
    *,
    preview_manifest_json: Path,
    preview_window_judge_jsonl: Path,
    render_contract_jsonl: Path,
) -> dict[str, Any]:
    preview_manifest = read_json(preview_manifest_json)
    preview_rows = [row for row in preview_manifest.get("rows") or [] if isinstance(row, dict)]
    judge_index = index_by_key(iter_jsonl(preview_window_judge_jsonl))
    contract_index = index_by_key(iter_jsonl(render_contract_jsonl))
    accepted_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for preview_row in sorted(preview_rows, key=row_key):
        judge_row = matching_judge_row(preview_row, judge_index)
        contract_row = matching_contract_row(preview_row, contract_index)
        reason = skip_reason(preview_row, judge_row=judge_row, contract_row=contract_row)
        if reason:
            skipped_rows.append(
                {
                    "track_id": str(preview_row.get("track_id") or ""),
                    "base_candidate_id": str(preview_row.get("base_candidate_id") or ""),
                    "alternate_candidate_id": str(preview_row.get("alternate_candidate_id") or ""),
                    "source_stem": normalize_source_stem(preview_row.get("source_stem")),
                    "window_start_sec": round(as_float(preview_row.get("window_start_sec")), 6),
                    "window_duration_sec": round(as_float(preview_row.get("window_duration_sec")), 6),
                    "skip_reason": reason,
                }
            )
            continue
        assert judge_row is not None
        assert contract_row is not None
        accepted_rows.append(accepted_row(preview_row, judge_row, contract_row))
    if not accepted_rows:
        return blocked_manifest(
            preview_manifest_json=preview_manifest_json,
            preview_window_judge_jsonl=preview_window_judge_jsonl,
            render_contract_jsonl=render_contract_jsonl,
            skipped_rows=skipped_rows,
        )
    return accepted_manifest(
        preview_manifest_json=preview_manifest_json,
        preview_window_judge_jsonl=preview_window_judge_jsonl,
        render_contract_jsonl=render_contract_jsonl,
        rows=accepted_rows,
        skipped_rows=skipped_rows,
    )


def build_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# No-Reference Segment Preview Listen Manifest Summary",
        "",
        f"- delivery_status: {manifest.get('delivery_status')}",
        f"- delivery_allowed: {str(manifest.get('delivery_allowed') is True).lower()}",
        f"- presentation_status: {manifest.get('presentation_status')}",
        f"- release_status: {manifest.get('release_status')}",
        f"- stem_type: {manifest.get('stem_type')}",
        f"- accepted_count: {manifest.get('accepted_count')}",
        f"- skipped_count: {manifest.get('skipped_count')}",
        f"- dominant_failure_reason: {manifest.get('dominant_failure_reason')}",
        f"- selected_vocal_path: {manifest.get('selected_vocal_path') or ''}",
        f"- selected_minus_path: {manifest.get('selected_minus_path') or ''}",
        "",
        "## Safety",
        "",
    ]
    safety = manifest.get("safety") if isinstance(manifest.get("safety"), dict) else {}
    for key in sorted(SAFETY_FLAGS):
        lines.append(f"- {key}: {str(safety.get(key) is True).lower()}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_no_reference_segment_preview_listen_manifest.py",
        entrypoint_path="/app/tools/build_no_reference_segment_preview_listen_manifest.py",
    )
    parser = argparse.ArgumentParser(description="Build a listen-preview manifest from post-render-improved segment previews.")
    parser.add_argument("--preview-manifest-json", type=Path, required=True)
    parser.add_argument("--preview-window-judge-jsonl", type=Path, required=True)
    parser.add_argument("--render-contract-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest(
        preview_manifest_json=args.preview_manifest_json,
        preview_window_judge_jsonl=args.preview_window_judge_jsonl,
        render_contract_jsonl=args.render_contract_jsonl,
    )
    write_json(args.output_manifest_json, manifest)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(build_summary(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                key: manifest.get(key)
                for key in (
                    "delivery_status",
                    "delivery_allowed",
                    "presentation_status",
                    "release_status",
                    "accepted_count",
                    "skipped_count",
                    "dominant_failure_reason",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest.get("delivery_allowed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
