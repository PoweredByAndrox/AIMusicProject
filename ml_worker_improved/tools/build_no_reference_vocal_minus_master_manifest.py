from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_status_contract import (  # noqa: E402
    ArtifactContractError,
    blocking_labels_from_values,
    can_promote_to_listen_result,
    sha256_file,
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
    "diagnostic_audio_exported": False,
    "audio_copied": False,
    "audio_exported": False,
    "release_ready": False,
}
PAIR_MIN_WINDOW_COUNT = 5


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
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
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() else "",
    }


def stable_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


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


def normalized_path_identity(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").lower()


def source_mix_path(manifest: dict[str, Any]) -> str:
    row = (manifest.get("accepted_listen_preview_rows") or [{}])[0]
    row_mix = row.get("mix_path") or row.get("source_audio_path") if isinstance(row, dict) else ""
    return str(manifest.get("mix_path") or manifest.get("source_audio_path") or row_mix or "")


def track_identity_match_method(vocal: dict[str, Any], minus: dict[str, Any]) -> str:
    vocal_track_id = str(vocal.get("track_id") or "")
    minus_track_id = str(minus.get("track_id") or "")
    if vocal_track_id and vocal_track_id == minus_track_id:
        return "exact_track_id"
    vocal_mix = normalized_path_identity(source_mix_path(vocal))
    minus_mix = normalized_path_identity(source_mix_path(minus))
    if vocal_mix and vocal_mix == minus_mix:
        return "shared_mix_path"
    return ""


def selected_vocal_path(manifest: dict[str, Any]) -> str:
    return str(manifest.get("selected_vocal_path") or manifest.get("vocal_path") or manifest.get("selected_audio_path") or "")


def selected_minus_path(manifest: dict[str, Any]) -> str:
    return str(
        manifest.get("selected_minus_path")
        or manifest.get("minus_path")
        or manifest.get("instrumental_path")
        or manifest.get("selected_audio_path")
        or ""
    )


def pair_vocal_path(row: dict[str, Any]) -> str:
    return str(row.get("vocal_path") or row.get("selected_vocal_path") or "")


def pair_minus_path(row: dict[str, Any]) -> str:
    return str(
        row.get("instrumental_path")
        or row.get("minus_path")
        or row.get("selected_minus_path")
        or row.get("selected_audio_path")
        or ""
    )


def pair_mix_path(row: dict[str, Any]) -> str:
    return str(row.get("mix_path") or row.get("source_audio_path") or "")


def paths_match(left: Any, right: Any) -> bool:
    return bool(normalized_path_identity(left) and normalized_path_identity(left) == normalized_path_identity(right))


def pair_row_matches_sources(row: dict[str, Any], vocal_manifest: dict[str, Any], minus_manifest: dict[str, Any]) -> bool:
    if not paths_match(pair_vocal_path(row), selected_vocal_path(vocal_manifest)):
        return False
    if not paths_match(pair_minus_path(row), selected_minus_path(minus_manifest)):
        return False

    row_track_id = str(row.get("track_id") or "")
    vocal_track_id = str(vocal_manifest.get("track_id") or "")
    minus_track_id = str(minus_manifest.get("track_id") or "")
    row_mix = pair_mix_path(row)
    source_mix = source_mix_path(vocal_manifest) or source_mix_path(minus_manifest)
    if row_mix and source_mix and not paths_match(row_mix, source_mix):
        return False
    if row_track_id and row_track_id not in {vocal_track_id, minus_track_id} and not paths_match(row_mix, source_mix):
        return False
    return True


def pair_row_failure_reason(row: dict[str, Any]) -> str:
    blocking_labels = blocking_labels_from_values(row.get("reasons"), row.get("decision"))
    if blocking_labels:
        return f"pair_compatibility_blocking_semantic_label:{blocking_labels[0]}"
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
    vocal_manifest: dict[str, Any],
    minus_manifest: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    if pair_compatibility_jsonl is None or not pair_compatibility_jsonl.exists():
        return None, "pair_compatibility_evidence_missing"
    matched_rows = [
        row
        for row in iter_jsonl(pair_compatibility_jsonl)
        if pair_row_matches_sources(row, vocal_manifest, minus_manifest)
    ]
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


def merged_safety(vocal: dict[str, Any], minus: dict[str, Any]) -> dict[str, bool]:
    safety = dict(SAFETY_FLAGS)
    for source in (vocal.get("safety"), minus.get("safety")):
        if not isinstance(source, dict):
            continue
        for key in safety:
            safety[key] = bool(safety[key] or source.get(key) is True)
    return safety


def source_safety_violations(vocal: dict[str, Any], minus: dict[str, Any]) -> list[str]:
    safety = merged_safety(vocal, minus)
    return [key for key, value in safety.items() if value is True]


def failure_reason(
    *,
    vocal_manifest_json: Path,
    minus_manifest_json: Path,
    vocal_manifest: dict[str, Any],
    minus_manifest: dict[str, Any],
) -> str:
    if str(vocal_manifest.get("stem_type") or "") != "vocal" or str(minus_manifest.get("stem_type") or "") != "minus":
        return "source_stem_type_mismatch"
    if source_safety_violations(vocal_manifest, minus_manifest):
        return "source_stem_safety_violation"
    if not source_manifest_accepted(vocal_manifest_json) or not source_manifest_accepted(minus_manifest_json):
        return "source_stem_manifest_not_accepted"
    if not source_manifest_listen_preview_only(vocal_manifest) or not source_manifest_listen_preview_only(minus_manifest):
        return "source_stem_not_listen_preview_only"
    if not track_identity_match_method(vocal_manifest, minus_manifest):
        return "track_id_mismatch"
    if not selected_vocal_path(vocal_manifest) or not selected_minus_path(minus_manifest):
        return "missing_selected_stem_path"
    return ""


def blocked_manifest(
    *,
    reason: str,
    vocal_manifest_json: Path,
    minus_manifest_json: Path,
    pair_compatibility_jsonl: Path | None = None,
    vocal_manifest: dict[str, Any],
    minus_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_type": "no_reference_vocal_minus_master_listen_preview",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": reason,
        "stem_type": "both",
        "track_id": str(vocal_manifest.get("track_id") or minus_manifest.get("track_id") or ""),
        "source_track_ids": {
            "vocal": str(vocal_manifest.get("track_id") or ""),
            "minus": str(minus_manifest.get("track_id") or ""),
        },
        "track_identity_match_method": track_identity_match_method(vocal_manifest, minus_manifest),
        "source_mix_path": source_mix_path(vocal_manifest) or source_mix_path(minus_manifest),
        "selected_vocal_path": selected_vocal_path(vocal_manifest),
        "selected_minus_path": selected_minus_path(minus_manifest),
        "pair_decision": "",
        "pair_compatibility_score": None,
        "source_pair_compatibility_jsonl": source_ref(pair_compatibility_jsonl) if pair_compatibility_jsonl is not None else {"path": "", "sha256": ""},
        "accepted_stem_count": 0,
        "source_vocal_manifest": source_ref(vocal_manifest_json),
        "source_minus_manifest": source_ref(minus_manifest_json),
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": merged_safety(vocal_manifest, minus_manifest),
    }


def accepted_manifest(
    *,
    vocal_manifest_json: Path,
    minus_manifest_json: Path,
    pair_compatibility_jsonl: Path,
    pair_compatibility_row: dict[str, Any],
    vocal_manifest: dict[str, Any],
    minus_manifest: dict[str, Any],
) -> dict[str, Any]:
    vocal_path = selected_vocal_path(vocal_manifest)
    minus_path = selected_minus_path(minus_manifest)
    track_id = str(vocal_manifest.get("track_id") or "")
    match_method = track_identity_match_method(vocal_manifest, minus_manifest)
    return {
        "manifest_type": "no_reference_vocal_minus_master_listen_preview",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": "both",
        "judge_mode": "no_reference_independent_vocal_and_minus_contract",
        "track_id": track_id,
        "source_track_ids": {
            "vocal": str(vocal_manifest.get("track_id") or ""),
            "minus": str(minus_manifest.get("track_id") or ""),
        },
        "track_identity_match_method": match_method,
        "source_mix_path": source_mix_path(vocal_manifest) or source_mix_path(minus_manifest),
        "selected_vocal_candidate_id": str(vocal_manifest.get("candidate_id") or ""),
        "selected_minus_candidate_id": str(minus_manifest.get("candidate_id") or ""),
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
        },
        "accepted_stem_count": 2,
        "accepted_listen_preview_rows": [
            {
                "track_id": track_id,
                "stem_type": "vocal",
                "candidate_id": str(vocal_manifest.get("candidate_id") or ""),
                "selected_audio_path": vocal_path,
                "source_manifest": str(vocal_manifest_json),
            },
            {
                "track_id": track_id,
                "stem_type": "minus",
                "candidate_id": str(minus_manifest.get("candidate_id") or ""),
                "selected_audio_path": minus_path,
                "source_manifest": str(minus_manifest_json),
            },
        ],
        "source_vocal_manifest": source_ref(vocal_manifest_json),
        "source_minus_manifest": source_ref(minus_manifest_json),
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
        "source_contracts": stable_unique([str(vocal_manifest_json), str(minus_manifest_json), str(pair_compatibility_jsonl)]),
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": merged_safety(vocal_manifest, minus_manifest),
    }


def build_manifest(
    *,
    vocal_manifest_json: Path,
    minus_manifest_json: Path,
    pair_compatibility_jsonl: Path | None = None,
) -> dict[str, Any]:
    vocal_manifest = read_json(vocal_manifest_json)
    minus_manifest = read_json(minus_manifest_json)
    reason = failure_reason(
        vocal_manifest_json=vocal_manifest_json,
        minus_manifest_json=minus_manifest_json,
        vocal_manifest=vocal_manifest,
        minus_manifest=minus_manifest,
    )
    if reason:
        return blocked_manifest(
            reason=reason,
            vocal_manifest_json=vocal_manifest_json,
            minus_manifest_json=minus_manifest_json,
            pair_compatibility_jsonl=pair_compatibility_jsonl,
            vocal_manifest=vocal_manifest,
            minus_manifest=minus_manifest,
        )
    pair_compatibility_row, pair_reason = select_pair_compatibility_row(
        pair_compatibility_jsonl=pair_compatibility_jsonl,
        vocal_manifest=vocal_manifest,
        minus_manifest=minus_manifest,
    )
    if pair_reason or pair_compatibility_row is None:
        return blocked_manifest(
            reason=pair_reason,
            vocal_manifest_json=vocal_manifest_json,
            minus_manifest_json=minus_manifest_json,
            pair_compatibility_jsonl=pair_compatibility_jsonl,
            vocal_manifest=vocal_manifest,
            minus_manifest=minus_manifest,
        )
    return accepted_manifest(
        vocal_manifest_json=vocal_manifest_json,
        minus_manifest_json=minus_manifest_json,
        pair_compatibility_jsonl=pair_compatibility_jsonl,
        pair_compatibility_row=pair_compatibility_row,
        vocal_manifest=vocal_manifest,
        minus_manifest=minus_manifest,
    )


def build_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# No-Reference Vocal Minus Master Summary",
        "",
        f"- delivery_status: {manifest.get('delivery_status')}",
        f"- delivery_allowed: {str(manifest.get('delivery_allowed') is True).lower()}",
        f"- presentation_status: {manifest.get('presentation_status')}",
        f"- release_status: {manifest.get('release_status')}",
        f"- stem_type: {manifest.get('stem_type')}",
        f"- accepted_stem_count: {manifest.get('accepted_stem_count')}",
        f"- pair_decision: {manifest.get('pair_decision')}",
        f"- pair_score_mode_decision: {manifest.get('pair_score_mode_decision', '')}",
        f"- pair_score_mode_reasons: {', '.join(str(item) for item in as_list(manifest.get('pair_score_mode_reasons')))}",
        f"- pair_compatibility_score: {manifest.get('pair_compatibility_score')}",
        f"- selected_vocal_path: {manifest.get('selected_vocal_path')}",
        f"- selected_minus_path: {manifest.get('selected_minus_path')}",
        f"- dominant_failure_reason: {manifest.get('dominant_failure_reason')}",
        "",
        "## Safety",
        "",
    ]
    safety = manifest.get("safety") if isinstance(manifest.get("safety"), dict) else {}
    for key in sorted(SAFETY_FLAGS):
        lines.append(f"- {key}: {str(safety.get(key) is True).lower()}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a no-reference master listen-preview manifest from accepted vocal and minus manifests.")
    parser.add_argument("--vocal-manifest-json", type=Path, required=True)
    parser.add_argument("--minus-manifest-json", type=Path, required=True)
    parser.add_argument("--pair-compatibility-jsonl", type=Path)
    parser.add_argument("--output-manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = build_manifest(
        vocal_manifest_json=args.vocal_manifest_json,
        minus_manifest_json=args.minus_manifest_json,
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
                    "accepted_stem_count",
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
