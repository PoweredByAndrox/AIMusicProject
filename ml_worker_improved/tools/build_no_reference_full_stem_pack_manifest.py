from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_status_contract import ArtifactContractError, can_promote_to_listen_result, sha256_file  # noqa: E402


REQUIRED_NONVOCAL_STEMS = ("bass", "drums", "melody", "other")
SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "release_ready": False,
    "audio_copied": False,
    "audio_exported": False,
    "diagnostic_audio_exported": False,
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path) if path.exists() else ""}


def as_bool(value: Any) -> bool:
    return value is True


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def stable_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def first_text(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        text = str(payload.get(key) or "").strip()
        if text:
            return text
    return ""


def merged_safety(*payloads: dict[str, Any]) -> dict[str, bool]:
    safety = dict(SAFETY_FLAGS)
    for payload in payloads:
        sources = [payload, payload.get("safety") if isinstance(payload.get("safety"), dict) else {}]
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key in safety:
                safety[key] = bool(safety[key] or source.get(key) is True)
    return safety


def safety_violations(safety: dict[str, bool]) -> list[str]:
    return [key for key in sorted(safety) if safety.get(key) is True]


def source_manifest_accepted(path: Path) -> bool:
    try:
        return can_promote_to_listen_result(path)
    except ArtifactContractError:
        return False


def source_master_listen_preview_only(master: dict[str, Any]) -> bool:
    return (
        str(master.get("delivery_status") or "") == "accepted"
        and master.get("delivery_allowed") is True
        and str(master.get("presentation_status") or "") == "accepted_for_listen_pack"
        and str(master.get("release_status") or "") == "listen_preview_only"
        and master.get("release_ready") is not True
        and master.get("can_mark_release_ready") is not True
        and master.get("can_publish_as_release") is not True
    )


def matrix_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows = matrix.get("ranked_candidates")
    if not isinstance(rows, list):
        rows = matrix.get("rows")
    return [row for row in rows or [] if isinstance(row, dict)]


def master_source_audio_path(master: dict[str, Any]) -> str:
    pair_compatibility = master.get("pair_compatibility") if isinstance(master.get("pair_compatibility"), dict) else {}
    return first_text(master, "source_mix_path", "source_audio_path") or first_text(pair_compatibility, "mix_path")


def row_source_listen_manifest_path(row: dict[str, Any]) -> str:
    manifest_ref = row.get("source_listen_manifest")
    if not isinstance(manifest_ref, dict):
        return ""
    return str(manifest_ref.get("path") or "").strip()


def row_source_audio_path(row: dict[str, Any]) -> str:
    direct = first_text(row, "source_audio_path", "audio_path", "source_mix_path")
    if direct:
        return direct
    manifest_path = row_source_listen_manifest_path(row)
    if not manifest_path:
        return ""
    try:
        manifest = read_json(Path(manifest_path))
    except (OSError, ValueError):
        return ""
    return first_text(manifest, "source_audio_path", "audio_path", "source_mix_path")


def row_track_identity_match(
    row: dict[str, Any],
    *,
    track_id: str,
    source_audio_path: str,
) -> tuple[bool, str]:
    row_track_id = str(row.get("track_id") or "")
    if not row_track_id or not track_id:
        return True, "missing_track_id_allowed"
    if row_track_id == track_id:
        return True, "exact_track_id"
    row_source_path = row_source_audio_path(row)
    if source_audio_path and row_source_path and row_source_path == source_audio_path:
        return True, "source_listen_manifest_source_audio_path"
    return False, "track_id_mismatch"


def row_release_safe(row: dict[str, Any]) -> bool:
    return (
        str(row.get("delivery_status") or "") in {"", "accepted"}
        and row.get("delivery_allowed") is True
        and str(row.get("presentation_status") or "") in {"", "accepted_for_listen_pack"}
        and str(row.get("release_status") or "listen_preview_only") == "listen_preview_only"
        and row.get("release_ready") is not True
        and row.get("can_mark_release_ready") is not True
        and row.get("can_publish_as_release") is not True
    )


def selected_stem_from_row(row: dict[str, Any], *, track_identity_match_method: str) -> dict[str, Any]:
    stem = str(row.get("stem_type") or "")
    return {
        "stem_type": stem,
        "path": str(row.get("selected_audio_path") or ""),
        "model_name": str(row.get("model_name") or ""),
        "semantic_quality_verified": row.get("semantic_quality_verified") is True,
        "stem_semantic_score": row.get("stem_semantic_score"),
        "track_id": str(row.get("track_id") or ""),
        "track_identity_match_method": track_identity_match_method,
        "source_listen_manifest": row.get("source_listen_manifest") if isinstance(row.get("source_listen_manifest"), dict) else {},
    }


def select_required_stem(
    *,
    stem_type: str,
    rows: Sequence[dict[str, Any]],
    track_id: str,
    source_audio_path: str,
) -> tuple[dict[str, Any] | None, str]:
    candidates = [row for row in rows if str(row.get("stem_type") or "") == stem_type]
    if not candidates:
        return None, f"missing_required_stem:{stem_type}"
    track_candidates: list[tuple[dict[str, Any], str]] = []
    for row in candidates:
        matches, match_method = row_track_identity_match(row, track_id=track_id, source_audio_path=source_audio_path)
        if matches:
            track_candidates.append((row, match_method))
    if not track_candidates:
        return None, f"track_id_mismatch:{stem_type}"
    for row, match_method in track_candidates:
        if not row_release_safe(row):
            continue
        if not str(row.get("selected_audio_path") or "") or row.get("selected_audio_exists") is False:
            continue
        if row.get("semantic_quality_verified") is not True:
            return None, f"stem_semantic_quality_not_verified:{stem_type}"
        blockers = [str(item) for item in as_list(row.get("stem_semantic_blockers")) if str(item)]
        if blockers:
            return None, f"stem_semantic_blocked:{stem_type}"
        safety = merged_safety(row)
        if safety_violations(safety):
            return None, f"source_stem_safety_violation:{stem_type}"
        return selected_stem_from_row(row, track_identity_match_method=match_method), ""
    return None, f"no_accepted_required_stem:{stem_type}"


def blocked_manifest(
    *,
    reason: str,
    master_manifest_json: Path,
    stem_score_matrix_json: Path,
    master: dict[str, Any],
    safety: dict[str, bool],
) -> dict[str, Any]:
    return {
        "manifest_type": "no_reference_full_stem_pack",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": reason,
        "stem_type": "full_stem_pack",
        "track_id": str(master.get("track_id") or ""),
        "selected_vocal_path": str(master.get("selected_vocal_path") or ""),
        "selected_minus_path": str(master.get("selected_minus_path") or ""),
        "selected_stems": {},
        "accepted_stem_count": 0,
        "source_master_manifest": source_ref(master_manifest_json),
        "source_stem_score_matrix": source_ref(stem_score_matrix_json),
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": safety,
    }


def accepted_manifest(
    *,
    master_manifest_json: Path,
    stem_score_matrix_json: Path,
    master: dict[str, Any],
    selected_nonvocal: dict[str, dict[str, Any]],
    safety: dict[str, bool],
) -> dict[str, Any]:
    selected_stems: dict[str, dict[str, Any]] = {
        "vocal": {
            "stem_type": "vocal",
            "path": str(master.get("selected_vocal_path") or ""),
            "source": "no_reference_vocal_minus_master_manifest",
        },
        "minus": {
            "stem_type": "minus",
            "path": str(master.get("selected_minus_path") or ""),
            "source": "no_reference_vocal_minus_master_manifest",
        },
    }
    selected_stems.update(selected_nonvocal)
    track_identity_methods = stable_unique(
        stem.get("track_identity_match_method") for stem in selected_nonvocal.values() if isinstance(stem, dict)
    )
    track_identity_match_method = (
        track_identity_methods[0] if len(track_identity_methods) == 1 else "mixed:" + ",".join(track_identity_methods)
    )
    return {
        "manifest_type": "no_reference_full_stem_pack",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": "full_stem_pack",
        "track_id": str(master.get("track_id") or ""),
        "source_mix_path": str(master.get("source_mix_path") or ""),
        "selected_vocal_path": selected_stems["vocal"]["path"],
        "selected_minus_path": selected_stems["minus"]["path"],
        "selected_stems": selected_stems,
        "accepted_stem_count": len(selected_stems),
        "required_stems": ["vocal", "minus", *REQUIRED_NONVOCAL_STEMS],
        "pair_decision": str(master.get("pair_decision") or ""),
        "pair_compatibility_score": master.get("pair_compatibility_score"),
        "pair_window_count": int(master.get("pair_window_count") or 0),
        "track_identity_match_method": track_identity_match_method,
        "track_identity_methods": track_identity_methods,
        "source_master_manifest": source_ref(master_manifest_json),
        "source_stem_score_matrix": source_ref(stem_score_matrix_json),
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": safety,
    }


def build_manifest(*, master_manifest_json: Path, stem_score_matrix_json: Path) -> dict[str, Any]:
    master = read_json(master_manifest_json)
    matrix = read_json(stem_score_matrix_json)
    safety = merged_safety(master)
    if str(master.get("stem_type") or "") != "both":
        return blocked_manifest(
            reason="master_stem_type_mismatch",
            master_manifest_json=master_manifest_json,
            stem_score_matrix_json=stem_score_matrix_json,
            master=master,
            safety=safety,
        )
    if safety_violations(safety):
        return blocked_manifest(
            reason="source_master_safety_violation",
            master_manifest_json=master_manifest_json,
            stem_score_matrix_json=stem_score_matrix_json,
            master=master,
            safety=safety,
        )
    if not source_manifest_accepted(master_manifest_json) or not source_master_listen_preview_only(master):
        return blocked_manifest(
            reason="master_manifest_not_accepted",
            master_manifest_json=master_manifest_json,
            stem_score_matrix_json=stem_score_matrix_json,
            master=master,
            safety=safety,
        )
    if not str(master.get("selected_vocal_path") or "") or not str(master.get("selected_minus_path") or ""):
        return blocked_manifest(
            reason="master_missing_vocal_or_minus_path",
            master_manifest_json=master_manifest_json,
            stem_score_matrix_json=stem_score_matrix_json,
            master=master,
            safety=safety,
        )

    selected_nonvocal: dict[str, dict[str, Any]] = {}
    rows = matrix_rows(matrix)
    track_id = str(master.get("track_id") or "")
    source_audio_path = master_source_audio_path(master)
    for stem in REQUIRED_NONVOCAL_STEMS:
        selected, reason = select_required_stem(
            stem_type=stem,
            rows=rows,
            track_id=track_id,
            source_audio_path=source_audio_path,
        )
        if reason or selected is None:
            return blocked_manifest(
                reason=reason,
                master_manifest_json=master_manifest_json,
                stem_score_matrix_json=stem_score_matrix_json,
                master=master,
                safety=safety,
            )
        selected_nonvocal[stem] = selected
        safety = merged_safety({"safety": safety}, selected)
    if safety_violations(safety):
        return blocked_manifest(
            reason="source_stem_safety_violation",
            master_manifest_json=master_manifest_json,
            stem_score_matrix_json=stem_score_matrix_json,
            master=master,
            safety=safety,
        )
    return accepted_manifest(
        master_manifest_json=master_manifest_json,
        stem_score_matrix_json=stem_score_matrix_json,
        master=master,
        selected_nonvocal=selected_nonvocal,
        safety=safety,
    )


def build_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# No-Reference Full Stem Pack Summary",
        "",
        f"- delivery_status: {manifest.get('delivery_status')}",
        f"- delivery_allowed: {str(manifest.get('delivery_allowed') is True).lower()}",
        f"- presentation_status: {manifest.get('presentation_status')}",
        f"- release_status: {manifest.get('release_status')}",
        f"- stem_type: {manifest.get('stem_type')}",
        f"- track_id: {manifest.get('track_id')}",
        f"- accepted_stem_count: {manifest.get('accepted_stem_count')}",
        f"- dominant_failure_reason: {manifest.get('dominant_failure_reason')}",
        "",
        "## Selected Stems",
        "",
    ]
    selected = manifest.get("selected_stems") if isinstance(manifest.get("selected_stems"), dict) else {}
    for stem in sorted(selected):
        entry = selected[stem] if isinstance(selected[stem], dict) else {}
        lines.append(f"- {stem}: {entry.get('path')}")
    lines.extend(["", "## Safety", ""])
    safety = manifest.get("safety") if isinstance(manifest.get("safety"), dict) else {}
    for key in sorted(SAFETY_FLAGS):
        lines.append(f"- {key}: {str(safety.get(key) is True).lower()}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a no-reference full stem pack manifest from accepted source manifests.")
    parser.add_argument("--master-manifest-json", type=Path, required=True)
    parser.add_argument("--stem-score-matrix-json", type=Path, required=True)
    parser.add_argument("--output-manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = build_manifest(
        master_manifest_json=args.master_manifest_json,
        stem_score_matrix_json=args.stem_score_matrix_json,
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
