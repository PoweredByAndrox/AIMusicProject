from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.artifact_status_contract import ArtifactContractError, can_promote_to_listen_result, sha256_file
from tools.docker_runtime import require_docker_runtime


SAFETY_FLAGS: dict[str, bool] = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "audio_copied": False,
    "audio_exported": False,
    "diagnostic_audio_exported": False,
    "release_ready": False,
}

BLOCKING_SAFETY_FLAGS = {
    "training_started",
    "fine_tune_started",
    "models_changed",
    "model_registry_edited",
    "cleanup_routes_added",
    "thresholds_relaxed",
    "audio_copied",
    "audio_exported",
    "diagnostic_audio_exported",
    "release_ready",
}

FORBIDDEN_ACTIONS = [
    "do_not_require_reference_stems",
    "do_not_require_reference_vocal_minus",
    "do_not_require_human_csv",
    "do_not_start_finetune",
    "do_not_train",
    "do_not_add_models",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_gates",
    "do_not_export_diagnostic_audio_as_result",
    "do_not_accept_best_of_bad",
    "do_not_mark_release_ready",
    "do_not_publish_as_final_result",
]


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest_must_be_object:{path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def source_ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path) if path.exists() else ""}


def short_id(*parts: Any) -> str:
    normalized = "|".join(str(part or "").strip().lower().replace("\\", "/") for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


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


def source_manifest_accepted(path: Path) -> bool:
    try:
        return can_promote_to_listen_result(path)
    except ArtifactContractError:
        return False


def source_manifest_listen_preview_only(manifest: dict[str, Any]) -> bool:
    return (
        str(manifest.get("delivery_status") or "") == "accepted"
        and manifest.get("delivery_allowed") is True
        and str(manifest.get("presentation_status") or "") == "accepted_for_listen_pack"
        and str(manifest.get("release_status") or "") == "listen_preview_only"
        and manifest.get("release_ready") is not True
        and manifest.get("can_mark_release_ready") is not True
        and manifest.get("can_publish_as_release") is not True
    )


def manifest_safety_flag(manifest: dict[str, Any], key: str) -> bool:
    safety = manifest.get("safety")
    if isinstance(safety, dict) and safety.get(key) is True:
        return True
    return manifest.get(key) is True


def safety_violations(manifest: dict[str, Any]) -> list[str]:
    return [key for key in sorted(BLOCKING_SAFETY_FLAGS) if manifest_safety_flag(manifest, key)]


def first_row(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = manifest.get("accepted_listen_preview_rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows[0]
    return {}


def manifest_vocal_path(manifest: dict[str, Any]) -> str:
    row = first_row(manifest)
    artifacts = manifest.get("selected_artifacts") if isinstance(manifest.get("selected_artifacts"), dict) else {}
    return str(
        manifest.get("selected_vocal_path")
        or artifacts.get("listen_segment_preview_vocal_flac")
        or artifacts.get("listen_selected_vocals_flac")
        or row.get("selected_vocal_path")
        or row.get("preview_vocal_path")
        or ""
    )


def manifest_minus_path(manifest: dict[str, Any]) -> str:
    row = first_row(manifest)
    artifacts = manifest.get("selected_artifacts") if isinstance(manifest.get("selected_artifacts"), dict) else {}
    return str(
        manifest.get("selected_minus_path")
        or artifacts.get("listen_segment_preview_minus_flac")
        or artifacts.get("listen_minus_flac")
        or row.get("selected_minus_path")
        or row.get("preview_minus_path")
        or ""
    )


def row_mix_path(row: dict[str, Any], manifest: dict[str, Any]) -> str:
    return str(row.get("mix_path") or row.get("source_audio_path") or manifest.get("mix_path") or manifest.get("source_audio_path") or "")


def path_exists(path_text: str) -> bool:
    return bool(path_text and Path(path_text).exists())


def source_failure_reason(*, manifest_json: Path, manifest: dict[str, Any]) -> str:
    if str(manifest.get("manifest_type") or "") != "no_reference_segment_preview_listen":
        return "source_manifest_type_mismatch"
    if not source_manifest_accepted(manifest_json):
        return "source_manifest_not_accepted"
    if not source_manifest_listen_preview_only(manifest):
        return "source_manifest_not_listen_preview_only"
    if safety_violations(manifest):
        return "source_manifest_safety_violation"
    if not isinstance(manifest.get("accepted_listen_preview_rows"), list) or not manifest.get("accepted_listen_preview_rows"):
        return "source_manifest_has_no_accepted_rows"
    return ""


def row_skip_reason(row: dict[str, Any], manifest: dict[str, Any]) -> str:
    mix_path = row_mix_path(row, manifest)
    vocal_path = str(row.get("selected_vocal_path") or row.get("preview_vocal_path") or manifest_vocal_path(manifest))
    minus_path = str(row.get("selected_minus_path") or row.get("preview_minus_path") or manifest_minus_path(manifest))
    if not mix_path:
        return "missing_source_mix_path"
    if not vocal_path:
        return "missing_selected_vocal_path"
    if not minus_path:
        return "missing_selected_minus_path"
    if not path_exists(mix_path):
        return "missing_source_mix_audio"
    if not path_exists(vocal_path):
        return "missing_selected_vocal_audio"
    if not path_exists(minus_path):
        return "missing_selected_minus_audio"
    if str(row.get("preview_decision") or "") != "preview_improved_window":
        return "source_preview_not_improved"
    if str(row.get("allowed_next_step") or "") != "expand_preview_policy":
        return "source_preview_not_approved_for_expansion"
    if as_float(row.get("score_delta")) <= 0.0:
        return "source_preview_score_delta_not_positive"
    return ""


def candidate_row(*, source_manifest_json: Path, manifest: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    mix_path = row_mix_path(row, manifest)
    vocal_path = str(row.get("selected_vocal_path") or row.get("preview_vocal_path") or manifest_vocal_path(manifest))
    minus_path = str(row.get("selected_minus_path") or row.get("preview_minus_path") or manifest_minus_path(manifest))
    track_id = str(row.get("track_id") or manifest.get("track_id") or "")
    start = round(as_float(row.get("window_start_sec")), 6)
    duration = round(as_float(row.get("window_duration_sec")), 6)
    candidate_id = "segment_preview_pair_" + short_id(track_id, mix_path, vocal_path, minus_path, start, duration)
    return {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": mix_path,
        "source_audio_path": mix_path,
        "vocal_path": vocal_path,
        "instrumental_path": minus_path,
        "minus_path": minus_path,
        "selected_vocal_path": vocal_path,
        "selected_minus_path": minus_path,
        "source_stem": normalize_source_stem(row.get("source_stem")),
        "base_candidate_id": str(row.get("base_candidate_id") or ""),
        "alternate_candidate_id": str(row.get("alternate_candidate_id") or ""),
        "window_start_sec": start,
        "window_duration_sec": duration,
        "base_window_score": as_float(row.get("base_window_score")),
        "preview_window_score": as_float(row.get("preview_window_score")),
        "score_delta": as_float(row.get("score_delta")),
        "preview_decision": str(row.get("preview_decision") or ""),
        "allowed_next_step": str(row.get("allowed_next_step") or ""),
        "source_segment_preview_manifest": str(source_manifest_json),
        "source_segment_preview_manifest_sha256": sha256_file(source_manifest_json),
        "candidate_family": "no_reference_segment_preview_pair",
        "candidate_origin": "accepted_segment_preview_listen_manifest_pair_bridge",
        "final_decision": "pair_candidate_ready_for_compatibility_judge",
        "pair_compatibility_decision": "",
        "requires_pair_compatibility_judge": True,
        "requires_dense_post_reconciliation_judge": True,
        "is_complete_pair": True,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "audio_copied": False,
        "audio_exported": False,
        "forbidden_actions": list(FORBIDDEN_ACTIONS),
    }


def blocked_manifest(
    *,
    reason: str,
    segment_preview_manifest_json: Path,
    manifest: dict[str, Any],
    skipped_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reason_counts = Counter(str(row.get("skip_reason") or "") for row in skipped_rows)
    return {
        "manifest_type": "no_reference_segment_preview_pair_candidate",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_pair_judge",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": reason,
        "stem_type": "both",
        "track_id": str(manifest.get("track_id") or ""),
        "selected_vocal_path": manifest_vocal_path(manifest),
        "selected_minus_path": manifest_minus_path(manifest),
        "candidate_count": 0,
        "candidate_rows": [],
        "skipped_rows": skipped_rows,
        "skip_reason_counts": dict(sorted(reason_counts.items())),
        "source_segment_preview_manifest": source_ref(segment_preview_manifest_json),
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": dict(SAFETY_FLAGS),
    }


def ready_manifest(
    *,
    segment_preview_manifest_json: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = rows[0]
    return {
        "manifest_type": "no_reference_segment_preview_pair_candidate",
        "delivery_status": "ready_for_pair_judge",
        "delivery_allowed": False,
        "presentation_status": "candidate_rows_ready_for_pair_judge",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": "both",
        "track_id": str(primary.get("track_id") or manifest.get("track_id") or ""),
        "source_mix_path": str(primary.get("mix_path") or ""),
        "selected_vocal_path": str(primary.get("selected_vocal_path") or ""),
        "selected_minus_path": str(primary.get("selected_minus_path") or ""),
        "candidate_count": len(rows),
        "candidate_rows": rows,
        "skipped_rows": skipped_rows,
        "source_segment_preview_manifest": source_ref(segment_preview_manifest_json),
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": dict(SAFETY_FLAGS),
    }


def build_pair_candidate_manifest(*, segment_preview_manifest_json: Path) -> dict[str, Any]:
    manifest = read_json(segment_preview_manifest_json)
    source_reason = source_failure_reason(manifest_json=segment_preview_manifest_json, manifest=manifest)
    if source_reason:
        return blocked_manifest(
            reason=source_reason,
            segment_preview_manifest_json=segment_preview_manifest_json,
            manifest=manifest,
            skipped_rows=[],
        )
    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for row in manifest.get("accepted_listen_preview_rows") or []:
        if not isinstance(row, dict):
            continue
        reason = row_skip_reason(row, manifest)
        if reason:
            skipped_rows.append(
                {
                    "track_id": str(row.get("track_id") or manifest.get("track_id") or ""),
                    "base_candidate_id": str(row.get("base_candidate_id") or ""),
                    "alternate_candidate_id": str(row.get("alternate_candidate_id") or ""),
                    "source_stem": normalize_source_stem(row.get("source_stem")),
                    "window_start_sec": round(as_float(row.get("window_start_sec")), 6),
                    "window_duration_sec": round(as_float(row.get("window_duration_sec")), 6),
                    "skip_reason": reason,
                }
            )
            continue
        rows.append(candidate_row(source_manifest_json=segment_preview_manifest_json, manifest=manifest, row=row))
    if not rows:
        return blocked_manifest(
            reason="no_pair_judge_ready_segment_preview_rows",
            segment_preview_manifest_json=segment_preview_manifest_json,
            manifest=manifest,
            skipped_rows=skipped_rows,
        )
    return ready_manifest(
        segment_preview_manifest_json=segment_preview_manifest_json,
        manifest=manifest,
        rows=rows,
        skipped_rows=skipped_rows,
    )


def build_summary_md(manifest: dict[str, Any]) -> str:
    lines = [
        "# No-Reference Segment Preview Pair Candidate",
        "",
        f"- delivery_status: {manifest.get('delivery_status')}",
        f"- delivery_allowed: {str(manifest.get('delivery_allowed') is True).lower()}",
        f"- presentation_status: {manifest.get('presentation_status')}",
        f"- release_status: {manifest.get('release_status')}",
        f"- dominant_failure_reason: {manifest.get('dominant_failure_reason')}",
        f"- stem_type: {manifest.get('stem_type')}",
        f"- candidate_count: {manifest.get('candidate_count')}",
        f"- release_ready: {str(manifest.get('release_ready') is True).lower()}",
        "",
        "## Safety",
        "",
    ]
    safety = manifest.get("safety") if isinstance(manifest.get("safety"), dict) else {}
    for key in sorted(SAFETY_FLAGS):
        lines.append(f"- {key}: {str(safety.get(key) is True).lower()}")
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    manifest: dict[str, Any],
    output_jsonl: Path,
    summary_json: Path,
    summary_md: Path,
) -> None:
    rows = manifest.get("candidate_rows") if isinstance(manifest.get("candidate_rows"), list) else []
    write_jsonl(output_jsonl, rows)
    write_json(summary_json, manifest)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(build_summary_md(manifest), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_no_reference_segment_preview_pair_candidate.py",
        entrypoint_path="/app/tools/build_no_reference_segment_preview_pair_candidate.py",
    )
    parser = argparse.ArgumentParser(
        description="Build pair-compatibility candidate rows from an accepted segment-preview listen manifest."
    )
    parser.add_argument("--segment-preview-manifest-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_pair_candidate_manifest(segment_preview_manifest_json=args.segment_preview_manifest_json)
    write_outputs(
        manifest=manifest,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        summary_md=args.summary_md,
    )
    print(
        json.dumps(
            {
                "delivery_status": manifest.get("delivery_status"),
                "dominant_failure_reason": manifest.get("dominant_failure_reason"),
                "candidate_count": manifest.get("candidate_count"),
                "output_jsonl": str(args.output_jsonl),
                "summary_json": str(args.summary_json),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest.get("delivery_status") == "ready_for_pair_judge" else 1


if __name__ == "__main__":
    raise SystemExit(main())
