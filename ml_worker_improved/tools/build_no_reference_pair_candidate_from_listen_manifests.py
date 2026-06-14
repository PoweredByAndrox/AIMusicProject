from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.artifact_status_contract import ArtifactContractError, can_promote_to_listen_result


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path) if path.exists() else ""}


def short_id(*parts: Any) -> str:
    normalized = "|".join(str(part or "").strip().lower().replace("\\", "/") for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def normalized_path_identity(path: str) -> str:
    return str(path or "").strip().replace("\\", "/").lower()


def first_row(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = manifest.get("accepted_listen_preview_rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows[0]
    return {}


def source_mix_path(manifest: dict[str, Any]) -> str:
    row = first_row(manifest)
    row_mix = str(row.get("mix_path") or row.get("source_audio_path") or "")
    return str(manifest.get("mix_path") or manifest.get("source_audio_path") or row_mix or "")


def selected_vocal_path(manifest: dict[str, Any]) -> str:
    artifacts = manifest.get("selected_artifacts") if isinstance(manifest.get("selected_artifacts"), dict) else {}
    row = first_row(manifest)
    return str(
        manifest.get("selected_vocal_path")
        or manifest.get("selected_audio_path")
        or artifacts.get("listen_vocals_flac")
        or row.get("selected_audio_path")
        or row.get("vocal_path")
        or ""
    )


def selected_minus_path(manifest: dict[str, Any]) -> str:
    artifacts = manifest.get("selected_artifacts") if isinstance(manifest.get("selected_artifacts"), dict) else {}
    row = first_row(manifest)
    return str(
        manifest.get("selected_minus_path")
        or manifest.get("selected_audio_path")
        or artifacts.get("listen_minus_flac")
        or row.get("selected_audio_path")
        or row.get("minus_path")
        or row.get("instrumental_path")
        or ""
    )


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


def track_identity_match_method(vocal_manifest: dict[str, Any], minus_manifest: dict[str, Any]) -> str:
    vocal_track_id = str(vocal_manifest.get("track_id") or "")
    minus_track_id = str(minus_manifest.get("track_id") or "")
    if vocal_track_id and vocal_track_id == minus_track_id:
        return "exact_track_id"
    vocal_mix = normalized_path_identity(source_mix_path(vocal_manifest))
    minus_mix = normalized_path_identity(source_mix_path(minus_manifest))
    if vocal_mix and vocal_mix == minus_mix:
        return "shared_mix_path"
    return ""


def manifest_safety_flag(manifest: dict[str, Any], key: str) -> bool:
    safety = manifest.get("safety")
    if isinstance(safety, dict) and safety.get(key) is True:
        return True
    return manifest.get(key) is True


def source_safety_violations(vocal_manifest: dict[str, Any], minus_manifest: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for key in sorted(BLOCKING_SAFETY_FLAGS):
        if manifest_safety_flag(vocal_manifest, key) or manifest_safety_flag(minus_manifest, key):
            violations.append(key)
    return violations


def merged_safety(vocal_manifest: dict[str, Any], minus_manifest: dict[str, Any]) -> dict[str, bool]:
    safety = dict(SAFETY_FLAGS)
    for key in safety:
        safety[key] = manifest_safety_flag(vocal_manifest, key) or manifest_safety_flag(minus_manifest, key)
    return safety


def source_manifest_accepted(path: Path) -> bool:
    try:
        return can_promote_to_listen_result(path)
    except ArtifactContractError:
        return False


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
    if not source_mix_path(vocal_manifest) and not source_mix_path(minus_manifest):
        return "missing_source_mix_path"
    return ""


def build_candidate_row(
    *,
    vocal_manifest_json: Path,
    minus_manifest_json: Path,
    vocal_manifest: dict[str, Any],
    minus_manifest: dict[str, Any],
) -> dict[str, Any]:
    vocal_path = selected_vocal_path(vocal_manifest)
    minus_path = selected_minus_path(minus_manifest)
    track_id = str(vocal_manifest.get("track_id") or minus_manifest.get("track_id") or "")
    mix_path = source_mix_path(vocal_manifest) or source_mix_path(minus_manifest)
    vocal_candidate_id = str(vocal_manifest.get("candidate_id") or "")
    minus_candidate_id = str(minus_manifest.get("candidate_id") or "")
    candidate_id = "listen_manifest_pair_" + short_id(track_id, mix_path, vocal_path, minus_path)
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
        "source_vocal_candidate_id": vocal_candidate_id,
        "source_minus_candidate_id": minus_candidate_id,
        "source_vocal_manifest": str(vocal_manifest_json),
        "source_minus_manifest": str(minus_manifest_json),
        "source_vocal_manifest_sha256": sha256_file(vocal_manifest_json),
        "source_minus_manifest_sha256": sha256_file(minus_manifest_json),
        "track_identity_match_method": track_identity_match_method(vocal_manifest, minus_manifest),
        "source_track_ids": {
            "vocal": str(vocal_manifest.get("track_id") or ""),
            "minus": str(minus_manifest.get("track_id") or ""),
        },
        "candidate_family": "listen_manifest_vocal_minus_pair",
        "candidate_origin": "accepted_listen_preview_manifest_pair_bridge",
        "final_decision": "pair_candidate_ready_for_compatibility_judge",
        "pair_compatibility_decision": "",
        "requires_pair_compatibility_judge": True,
        "requires_dense_post_reconciliation_judge": True,
        "is_complete_pair": True,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "audio_copied": False,
        "audio_exported": False,
        "forbidden_actions": [
            "do_not_require_reference_stems",
            "do_not_require_human_csv",
            "do_not_start_finetune",
            "do_not_train",
            "do_not_add_models",
            "do_not_add_cleanup_debleed_denoise",
            "do_not_relax_gates",
            "do_not_export_diagnostic_audio_as_result",
            "do_not_accept_best_of_bad",
        ],
    }


def blocked_manifest(
    *,
    reason: str,
    vocal_manifest_json: Path,
    minus_manifest_json: Path,
    vocal_manifest: dict[str, Any],
    minus_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_type": "no_reference_pair_candidate_from_listen_manifests",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_pair_judge",
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
        "candidate_count": 0,
        "candidate_rows": [],
        "source_vocal_manifest": source_ref(vocal_manifest_json),
        "source_minus_manifest": source_ref(minus_manifest_json),
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": merged_safety(vocal_manifest, minus_manifest),
    }


def ready_manifest(
    *,
    vocal_manifest_json: Path,
    minus_manifest_json: Path,
    vocal_manifest: dict[str, Any],
    minus_manifest: dict[str, Any],
) -> dict[str, Any]:
    row = build_candidate_row(
        vocal_manifest_json=vocal_manifest_json,
        minus_manifest_json=minus_manifest_json,
        vocal_manifest=vocal_manifest,
        minus_manifest=minus_manifest,
    )
    return {
        "manifest_type": "no_reference_pair_candidate_from_listen_manifests",
        "delivery_status": "ready_for_pair_judge",
        "delivery_allowed": False,
        "presentation_status": "candidate_rows_ready_for_pair_judge",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": "both",
        "track_id": row["track_id"],
        "source_track_ids": row["source_track_ids"],
        "track_identity_match_method": row["track_identity_match_method"],
        "source_mix_path": row["mix_path"],
        "selected_vocal_path": row["selected_vocal_path"],
        "selected_minus_path": row["selected_minus_path"],
        "candidate_count": 1,
        "candidate_rows": [row],
        "source_vocal_manifest": source_ref(vocal_manifest_json),
        "source_minus_manifest": source_ref(minus_manifest_json),
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": merged_safety(vocal_manifest, minus_manifest),
    }


def build_pair_candidate_manifest(
    *,
    vocal_manifest_json: Path,
    minus_manifest_json: Path,
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
            vocal_manifest=vocal_manifest,
            minus_manifest=minus_manifest,
        )
    return ready_manifest(
        vocal_manifest_json=vocal_manifest_json,
        minus_manifest_json=minus_manifest_json,
        vocal_manifest=vocal_manifest,
        minus_manifest=minus_manifest,
    )


def build_summary_md(manifest: dict[str, Any]) -> str:
    lines = [
        "# No-Reference Pair Candidate From Listen Manifests",
        "",
        f"- delivery_status: {manifest.get('delivery_status')}",
        f"- delivery_allowed: {str(manifest.get('delivery_allowed') is True).lower()}",
        f"- presentation_status: {manifest.get('presentation_status')}",
        f"- release_status: {manifest.get('release_status')}",
        f"- dominant_failure_reason: {manifest.get('dominant_failure_reason')}",
        f"- stem_type: {manifest.get('stem_type')}",
        f"- candidate_count: {manifest.get('candidate_count')}",
        f"- track_identity_match_method: {manifest.get('track_identity_match_method')}",
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
    parser = argparse.ArgumentParser(
        description="Build a pair-compatibility candidate row from accepted vocal and minus listen-preview manifests."
    )
    parser.add_argument("--vocal-manifest-json", type=Path, required=True)
    parser.add_argument("--minus-manifest-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = build_pair_candidate_manifest(
        vocal_manifest_json=args.vocal_manifest_json,
        minus_manifest_json=args.minus_manifest_json,
    )
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
                "track_identity_match_method": manifest.get("track_identity_match_method"),
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
