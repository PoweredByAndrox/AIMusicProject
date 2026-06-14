from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.artifact_status_contract import ArtifactContractError, can_promote_to_listen_result, sha256_file
from tools.docker_runtime import require_docker_runtime


PAIR_MIN_WINDOW_COUNT = 5

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


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest_must_be_object:{path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path) if path.exists() else ""}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalized_path_identity(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").lower()


def paths_match(left: Any, right: Any) -> bool:
    return bool(normalized_path_identity(left) and normalized_path_identity(left) == normalized_path_identity(right))


def first_row(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = manifest.get("accepted_listen_preview_rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows[0]
    return {}


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


def source_mix_path(manifest: dict[str, Any]) -> str:
    row = first_row(manifest)
    return str(manifest.get("mix_path") or manifest.get("source_audio_path") or row.get("mix_path") or row.get("source_audio_path") or "")


def selected_vocal_path(manifest: dict[str, Any]) -> str:
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


def selected_minus_path(manifest: dict[str, Any]) -> str:
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


def pair_vocal_path(row: dict[str, Any]) -> str:
    return str(row.get("vocal_path") or row.get("selected_vocal_path") or "")


def pair_minus_path(row: dict[str, Any]) -> str:
    return str(row.get("instrumental_path") or row.get("minus_path") or row.get("selected_minus_path") or "")


def pair_mix_path(row: dict[str, Any]) -> str:
    return str(row.get("mix_path") or row.get("source_audio_path") or "")


def source_failure_reason(*, segment_preview_manifest_json: Path, manifest: dict[str, Any]) -> str:
    if str(manifest.get("manifest_type") or "") != "no_reference_segment_preview_listen":
        return "source_manifest_type_mismatch"
    if not source_manifest_accepted(segment_preview_manifest_json):
        return "source_manifest_not_accepted"
    if not source_manifest_listen_preview_only(manifest):
        return "source_manifest_not_listen_preview_only"
    if safety_violations(manifest):
        return "source_manifest_safety_violation"
    if not source_mix_path(manifest):
        return "source_manifest_missing_mix_path"
    if not selected_vocal_path(manifest) or not selected_minus_path(manifest):
        return "source_manifest_missing_selected_paths"
    return ""


def pair_row_matches_source(row: dict[str, Any], manifest: dict[str, Any]) -> bool:
    if not paths_match(pair_vocal_path(row), selected_vocal_path(manifest)):
        return False
    if not paths_match(pair_minus_path(row), selected_minus_path(manifest)):
        return False
    row_mix = pair_mix_path(row)
    mix = source_mix_path(manifest)
    if row_mix and mix and not paths_match(row_mix, mix):
        return False
    row_track = str(row.get("track_id") or "")
    source_track = str(manifest.get("track_id") or "")
    if row_track and source_track and row_track != source_track and not paths_match(row_mix, mix):
        return False
    return True


def pair_row_failure_reason(row: dict[str, Any]) -> str:
    if str(row.get("score_mode") or "") != "both":
        return "pair_compatibility_not_both_mode"
    if str(row.get("pair_decision") or "") != "pair_compatible":
        return "pair_compatibility_not_accepted"
    if as_list(row.get("pair_blockers")):
        return "pair_compatibility_blockers_present"
    try:
        window_count = int(row.get("window_count") or 0)
    except (TypeError, ValueError):
        window_count = 0
    if window_count < PAIR_MIN_WINDOW_COUNT:
        return "pair_compatibility_insufficient_window_count"
    if row.get("pair_compatibility_score") is None:
        return "pair_compatibility_not_accepted"
    return ""


def select_pair_compatibility_row(
    *,
    pair_compatibility_jsonl: Path | None,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    if pair_compatibility_jsonl is None or not pair_compatibility_jsonl.exists():
        return None, "pair_compatibility_evidence_missing"
    matched_rows = [row for row in iter_jsonl(pair_compatibility_jsonl) if pair_row_matches_source(row, manifest)]
    if not matched_rows:
        return None, "pair_compatibility_no_exact_match"
    first_failure = ""
    for row in matched_rows:
        reason = pair_row_failure_reason(row)
        if not reason:
            return row, ""
        if not first_failure:
            first_failure = reason
    return None, first_failure or "pair_compatibility_not_accepted"


def blocked_manifest(
    *,
    reason: str,
    segment_preview_manifest_json: Path,
    pair_compatibility_jsonl: Path | None,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_type": "no_reference_segment_preview_pair_listen",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": reason,
        "stem_type": "both",
        "track_id": str(manifest.get("track_id") or ""),
        "source_mix_path": source_mix_path(manifest),
        "selected_vocal_path": selected_vocal_path(manifest),
        "selected_minus_path": selected_minus_path(manifest),
        "pair_decision": "",
        "pair_compatibility_score": None,
        "accepted_stem_count": 0,
        "source_segment_preview_manifest": source_ref(segment_preview_manifest_json),
        "source_pair_compatibility_jsonl": source_ref(pair_compatibility_jsonl) if pair_compatibility_jsonl is not None else {"path": "", "sha256": ""},
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": dict(SAFETY_FLAGS),
    }


def accepted_manifest(
    *,
    segment_preview_manifest_json: Path,
    pair_compatibility_jsonl: Path,
    pair_compatibility_row: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    vocal_path = selected_vocal_path(manifest)
    minus_path = selected_minus_path(manifest)
    mix_path = source_mix_path(manifest)
    track_id = str(manifest.get("track_id") or pair_compatibility_row.get("track_id") or "")
    return {
        "manifest_type": "no_reference_segment_preview_pair_listen",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": "both",
        "judge_mode": "no_reference_segment_preview_pair_contract",
        "track_id": track_id,
        "source_mix_path": mix_path,
        "mix_path": mix_path,
        "selected_vocal_candidate_id": str(manifest.get("candidate_id") or ""),
        "selected_minus_candidate_id": str(manifest.get("candidate_id") or ""),
        "pair_candidate_id": str(pair_compatibility_row.get("candidate_id") or ""),
        "pair_decision": str(pair_compatibility_row.get("pair_decision") or ""),
        "pair_score_mode_decision": str(pair_compatibility_row.get("decision") or ""),
        "pair_score_mode_reasons": as_list(pair_compatibility_row.get("reasons")),
        "pair_compatibility_score": float(pair_compatibility_row.get("pair_compatibility_score") or 0.0),
        "pair_blockers": as_list(pair_compatibility_row.get("pair_blockers")),
        "pair_window_count": int(pair_compatibility_row.get("window_count") or 0),
        "selected_vocal_path": vocal_path,
        "selected_minus_path": minus_path,
        "selected_audio_path": vocal_path,
        "selected_artifacts": {
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "listen_vocals_flac": vocal_path,
            "listen_minus_flac": minus_path,
            "listen_segment_preview_vocal_flac": vocal_path,
            "listen_segment_preview_minus_flac": minus_path,
        },
        "accepted_stem_count": 2,
        "accepted_listen_preview_rows": [
            {
                "track_id": track_id,
                "stem_type": "vocal",
                "candidate_id": str(manifest.get("candidate_id") or ""),
                "selected_audio_path": vocal_path,
                "source_manifest": str(segment_preview_manifest_json),
            },
            {
                "track_id": track_id,
                "stem_type": "minus",
                "candidate_id": str(manifest.get("candidate_id") or ""),
                "selected_audio_path": minus_path,
                "source_manifest": str(segment_preview_manifest_json),
            },
        ],
        "source_segment_preview_manifest": source_ref(segment_preview_manifest_json),
        "source_pair_compatibility_jsonl": source_ref(pair_compatibility_jsonl),
        "pair_compatibility": {
            "source_jsonl": str(pair_compatibility_jsonl),
            "candidate_id": str(pair_compatibility_row.get("candidate_id") or ""),
            "score_mode": str(pair_compatibility_row.get("score_mode") or ""),
            "decision": str(pair_compatibility_row.get("decision") or ""),
            "reasons": as_list(pair_compatibility_row.get("reasons")),
            "pair_decision": str(pair_compatibility_row.get("pair_decision") or ""),
            "pair_compatibility_score": float(pair_compatibility_row.get("pair_compatibility_score") or 0.0),
            "pair_blockers": as_list(pair_compatibility_row.get("pair_blockers")),
            "window_count": int(pair_compatibility_row.get("window_count") or 0),
            "vocal_path": pair_vocal_path(pair_compatibility_row),
            "minus_path": pair_minus_path(pair_compatibility_row),
            "mix_path": pair_mix_path(pair_compatibility_row),
        },
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(
    *,
    segment_preview_manifest_json: Path,
    pair_compatibility_jsonl: Path | None = None,
) -> dict[str, Any]:
    manifest = read_json(segment_preview_manifest_json)
    reason = source_failure_reason(segment_preview_manifest_json=segment_preview_manifest_json, manifest=manifest)
    if reason:
        return blocked_manifest(
            reason=reason,
            segment_preview_manifest_json=segment_preview_manifest_json,
            pair_compatibility_jsonl=pair_compatibility_jsonl,
            manifest=manifest,
        )
    pair_row, pair_reason = select_pair_compatibility_row(pair_compatibility_jsonl=pair_compatibility_jsonl, manifest=manifest)
    if pair_reason or pair_row is None:
        return blocked_manifest(
            reason=pair_reason,
            segment_preview_manifest_json=segment_preview_manifest_json,
            pair_compatibility_jsonl=pair_compatibility_jsonl,
            manifest=manifest,
        )
    assert pair_compatibility_jsonl is not None
    return accepted_manifest(
        segment_preview_manifest_json=segment_preview_manifest_json,
        pair_compatibility_jsonl=pair_compatibility_jsonl,
        pair_compatibility_row=pair_row,
        manifest=manifest,
    )


def build_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# No-Reference Segment Preview Pair Listen Manifest",
        "",
        f"- delivery_status: {manifest.get('delivery_status')}",
        f"- delivery_allowed: {str(manifest.get('delivery_allowed') is True).lower()}",
        f"- presentation_status: {manifest.get('presentation_status')}",
        f"- release_status: {manifest.get('release_status')}",
        f"- dominant_failure_reason: {manifest.get('dominant_failure_reason')}",
        f"- stem_type: {manifest.get('stem_type')}",
        f"- accepted_stem_count: {manifest.get('accepted_stem_count')}",
        f"- pair_decision: {manifest.get('pair_decision')}",
        f"- pair_compatibility_score: {manifest.get('pair_compatibility_score')}",
        f"- release_ready: {str(manifest.get('release_ready') is True).lower()}",
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
        tool_name="build_no_reference_segment_preview_pair_listen_manifest.py",
        entrypoint_path="/app/tools/build_no_reference_segment_preview_pair_listen_manifest.py",
    )
    parser = argparse.ArgumentParser(
        description="Build a listen-preview pair manifest from an accepted segment preview and exact both-mode judge row."
    )
    parser.add_argument("--segment-preview-manifest-json", type=Path, required=True)
    parser.add_argument("--pair-compatibility-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest(
        segment_preview_manifest_json=args.segment_preview_manifest_json,
        pair_compatibility_jsonl=args.pair_compatibility_jsonl,
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
                    "dominant_failure_reason",
                    "pair_decision",
                    "pair_compatibility_score",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest.get("delivery_allowed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
