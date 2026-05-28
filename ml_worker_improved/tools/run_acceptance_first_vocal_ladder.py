from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from tools.artifact_status_contract import ArtifactContractError, read_manifest, status_view_from_manifest
from worker.model_registry import default_model_registry


FORBIDDEN_ACTIONS_ENFORCED = [
    "no_dsp_repair_loop",
    "no_cleanup_debleed_denoise_escalation",
    "no_best_of_bad_export",
]


def candidate_model_ids(payload: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("model_id", "model_name", "model", "registered_model_id", "source_model_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            ids.add(value.strip())
    for key in ("candidate", "selected_candidate", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            ids.update(candidate_model_ids(value))
    return ids


def registry_identifiers() -> set[str]:
    registry = default_model_registry()
    identifiers: set[str] = set()
    entries = getattr(registry, "entries", ())
    if isinstance(entries, dict):
        entries = tuple(entries.values())
    for entry in entries:
        model_id = getattr(entry, "model_id", None)
        if model_id:
            identifiers.add(str(model_id))
        aliases = getattr(entry, "aliases", ()) or ()
        identifiers.update(str(alias) for alias in aliases)
    return identifiers


def validate_registered_candidate(manifest_path: Path) -> tuple[bool, str]:
    payload = read_manifest(manifest_path)
    model_ids = candidate_model_ids(payload)
    if not model_ids:
        return False, "missing_model_id"
    registered = registry_identifiers()
    if not registered:
        return False, "model_registry_unavailable"
    unknown = sorted(model_ids - registered)
    if unknown:
        return False, f"unregistered_model_id:{','.join(unknown)}"
    return True, "registered"


def choose_candidate(candidate_manifests: list[Path]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    evaluated: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for path in candidate_manifests:
        ok, reason = validate_registered_candidate(path)
        view = None
        promotable = False
        if ok:
            view = status_view_from_manifest(path)
            try:
                view.assert_promotable()
                promotable = True
            except ArtifactContractError:
                promotable = False
        record: dict[str, Any] = {
            "candidate_manifest": str(path),
            "registered": ok,
            "registration_reason": reason,
            "promotable": promotable,
        }
        if view is not None:
            record.update(
                {
                    "delivery_status": view.delivery_status,
                    "presentation_status": view.presentation_status,
                    "release_status": view.release_status,
                    "delivery_allowed": view.delivery_allowed,
                    "dominant_failure_reason": view.dominant_failure_reason,
                    "audio_path": view.audio_path,
                    "model_id": view.model_id,
                    "candidate_id": view.candidate_id,
                    "source_manifest": view.source_manifest,
                    "manifest_sha256": view.manifest_sha256,
                }
            )
        evaluated.append(record)
        if selected is None and promotable:
            selected = record
    return selected, evaluated


def dominant_failure_from_evaluated(evaluated: list[dict[str, Any]]) -> str:
    for record in evaluated:
        reason = str(record.get("dominant_failure_reason") or "").strip()
        if reason and reason not in {"none", "unknown"}:
            return reason
        registration_reason = str(record.get("registration_reason") or "").strip()
        if registration_reason and registration_reason != "registered":
            return registration_reason
    return "no_accepted_candidate"


def maybe_export_selected_audio(selected: dict[str, Any], listen_dir: Path) -> str | None:
    audio_path = selected.get("audio_path")
    if not audio_path:
        return None
    source = Path(str(audio_path))
    if not source.exists():
        return None
    listen_dir.mkdir(parents=True, exist_ok=True)
    target = listen_dir / source.name
    shutil.copy2(source, target)
    return str(target)


def build_final_manifest(candidate_manifests: list[Path], listen_dir: Path | None = None) -> dict[str, Any]:
    selected, evaluated = choose_candidate(candidate_manifests)
    if selected is None:
        return {
            "schema": "acceptance_first_vocal_ladder.v1",
            "status": "no_accepted_candidate",
            "selected_candidate": None,
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "dominant_failure_reason": dominant_failure_from_evaluated(evaluated),
            "forbidden_actions_enforced": FORBIDDEN_ACTIONS_ENFORCED,
            "evaluated_candidates": evaluated,
        }
    exported = maybe_export_selected_audio(selected, listen_dir) if listen_dir else None
    return {
        "schema": "acceptance_first_vocal_ladder.v1",
        "status": "accepted",
        "selected_candidate": selected.get("candidate_id") or selected.get("candidate_manifest"),
        "selected_candidate_manifest": selected.get("candidate_manifest"),
        "delivery_allowed": True,
        "presentation_status": selected.get("presentation_status"),
        "release_status": selected.get("release_status"),
        "dominant_failure_reason": "none",
        "exported_listen_audio": exported,
        "forbidden_actions_enforced": FORBIDDEN_ACTIONS_ENFORCED,
        "evaluated_candidates": evaluated,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--listen-dir", type=Path)
    args = parser.parse_args()

    manifest = build_final_manifest(args.candidate_manifest, args.listen_dir)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
