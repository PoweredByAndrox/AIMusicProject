#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_testing_stuff_music_corpus_vocal_judge_inputs import (  # noqa: E402
    SAFETY_FLAGS,
    collect_completed_candidates,
    source_ref,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime  # noqa: E402

CANDIDATES_JSONL_NAME = "no_reference_minus_windowed_judge_candidates.jsonl"
MANIFEST_JSON_NAME = "no_reference_minus_windowed_judge_candidates_manifest.json"
SUMMARY_MD_NAME = "no_reference_minus_windowed_judge_candidates_summary.md"

FORBIDDEN_ACTIONS = ["present_as_listen_result", "mark_release_ready", "start_training"]
MINUS_TARGET_STEMS = {"minus", "instrumental"}
VOCAL_TARGET_STEMS = {"vocal", "vocals"}
FORBIDDEN_MINUS_TARGET_SOURCES = {
    "sum_non_vocal_stems",
    "rebuilt_non_vocal_stems",
    "stage1_non_vocal_sum",
}
INSTRUMENTAL_MINUS_NAMING_CONTRACT = "source_manifest_uses_instrumental_stem; judge_rows_use_minus_product_role"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def manifest_stem_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stems = manifest.get("stems")
    if not isinstance(stems, dict):
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for stem, value in stems.items():
        if isinstance(value, dict) and value.get("path"):
            entries[str(stem)] = dict(value)
            entries[str(stem)]["path"] = str(value["path"])
    return entries


def manifest_stem_paths(manifest: dict[str, Any]) -> dict[str, str]:
    return {stem: str(entry["path"]) for stem, entry in manifest_stem_entries(manifest).items()}


def explicit_stem_entry(manifest: dict[str, Any], names: Sequence[str]) -> dict[str, Any]:
    entries = manifest_stem_entries(manifest)
    for name in names:
        entry = entries.get(name)
        if entry:
            return entry
    return {}


def explicit_vocal_entry(manifest: dict[str, Any]) -> dict[str, Any]:
    return explicit_stem_entry(manifest, ("vocals", "vocal"))


def vocal_entry_role(*, target_stem: str, entry: dict[str, Any]) -> str:
    if not entry:
        return ""
    if entry.get("residual_companion") is True:
        return "residual_companion"
    if target_stem in VOCAL_TARGET_STEMS:
        return "selected_vocal_candidate"
    return "minus_route_vocal_companion"


def is_selected_vocal_role(role: str) -> bool:
    return role == "selected_vocal_candidate"


def vocal_entry_payload(*, path: str, role: str, source: str = "", model_id: str = "") -> dict[str, Any]:
    return {
        "path": str(path),
        "role": str(role),
        "source": str(source),
        "model_id": str(model_id),
        "is_selected_vocal_candidate": is_selected_vocal_role(role),
    }


def explicit_minus_path(stem_paths: dict[str, str]) -> str:
    return stem_paths.get("minus") or stem_paths.get("instrumental") or ""


def explicit_vocal_path(stem_paths: dict[str, str]) -> str:
    return stem_paths.get("vocals") or stem_paths.get("vocal") or ""


def manifest_target_source(manifest: dict[str, Any]) -> str:
    target_artifact = manifest.get("target_stem_artifact")
    if isinstance(target_artifact, dict) and target_artifact.get("source"):
        return str(target_artifact["source"])
    return str(manifest.get("target_source") or "")


def source_manifest_target_stem(candidate: dict[str, Any], manifest: dict[str, Any]) -> str:
    return str(manifest.get("target_stem") or candidate.get("target_stem") or "").strip().lower()


def selected_minus_manifest_stem_name(*, source_target_stem: str, stem_paths: dict[str, str], minus_path: str) -> str:
    if source_target_stem in MINUS_TARGET_STEMS and stem_paths.get(source_target_stem):
        return source_target_stem
    normalized_minus_path = str(minus_path or "").replace("\\", "/").lower()
    for stem_name in ("minus", "instrumental"):
        stem_path = str(stem_paths.get(stem_name) or "").replace("\\", "/").lower()
        if stem_path and normalized_minus_path and stem_path == normalized_minus_path:
            return stem_name
    if source_target_stem in MINUS_TARGET_STEMS:
        return source_target_stem
    return "minus"


def selected_minus_entry(manifest: dict[str, Any], *, minus_path: str) -> dict[str, Any]:
    entries = manifest_stem_entries(manifest)
    normalized_minus_path = str(minus_path or "").replace("\\", "/").lower()
    for stem_name in ("minus", "instrumental"):
        entry = entries.get(stem_name)
        stem_path = str((entry or {}).get("path") or "").replace("\\", "/").lower()
        if stem_path and normalized_minus_path and stem_path == normalized_minus_path:
            return dict(entry)
    for stem_name in ("minus", "instrumental"):
        entry = entries.get(stem_name)
        if entry:
            return dict(entry)
    return {}


def role_safe_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("role_safe_stem_contract")
    return value if isinstance(value, dict) else {}


def target_stem_artifact(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("target_stem_artifact")
    return value if isinstance(value, dict) else {}


def minus_role_metadata(
    *,
    candidate: dict[str, Any],
    manifest: dict[str, Any],
    stem_paths: dict[str, str],
    minus_path: str,
) -> dict[str, str]:
    source_target_stem = source_manifest_target_stem(candidate, manifest)
    selected_entry = selected_minus_entry(manifest, minus_path=minus_path)
    contract = role_safe_contract(manifest)
    target_artifact = target_stem_artifact(manifest)
    manifest_stem_name = selected_minus_manifest_stem_name(
        source_target_stem=source_target_stem,
        stem_paths=stem_paths,
        minus_path=minus_path,
    )
    manifest_stem_name = str(
        selected_entry.get("manifest_stem_name")
        or contract.get("manifest_stem_name")
        or target_artifact.get("manifest_stem_name")
        or manifest_stem_name
    )
    canonical_stem_type = str(
        selected_entry.get("canonical_stem_type")
        or contract.get("canonical_stem_type")
        or target_artifact.get("canonical_stem_type")
        or (source_target_stem if source_target_stem in MINUS_TARGET_STEMS else manifest_stem_name)
    )
    model_output_stem_type = str(
        selected_entry.get("model_output_stem_type")
        or contract.get("model_output_stem_type")
        or target_artifact.get("model_output_stem_type")
        or manifest_stem_name
        or canonical_stem_type
    )
    naming_contract = (
        INSTRUMENTAL_MINUS_NAMING_CONTRACT
        if model_output_stem_type == "instrumental" or source_target_stem == "instrumental"
        else str(selected_entry.get("naming_contract") or "source_manifest_audio_stem_mapped_to_minus_product_role")
    )
    product_audio_path = str(
        selected_entry.get("product_audio_path") or manifest.get("target_product_audio_path") or minus_path
    )
    model_output_audio_path = str(
        selected_entry.get("model_output_audio_path")
        or target_artifact.get("model_output_audio_path")
        or manifest.get("target_model_output_audio_path")
        or manifest.get("target_audio_path")
        or minus_path
    )
    return {
        "source_manifest_target_stem": source_target_stem,
        "canonical_stem_type": canonical_stem_type,
        "user_facing_stem_role": str(
            selected_entry.get("user_facing_stem_role")
            or contract.get("user_facing_stem_role")
            or target_artifact.get("user_facing_stem_role")
            or "minus"
        ),
        "manifest_stem_name": manifest_stem_name,
        "product_stem_type": str(
            selected_entry.get("product_stem_type")
            or contract.get("product_stem_type")
            or target_artifact.get("product_stem_type")
            or "minus"
        ),
        "judge_stem_type": "minus",
        "stem_type_source": "judge_product_role",
        "model_output_stem_type": model_output_stem_type,
        "source_manifest_stem_type": source_target_stem,
        "naming_contract": naming_contract,
        "product_audio_path": product_audio_path,
        "model_output_audio_path": model_output_audio_path,
    }


def candidate_target_source(candidate: dict[str, Any], manifest: dict[str, Any]) -> str:
    execution_result = candidate.get("execution_result")
    execution_candidate = (
        execution_result.get("candidate_manifest")
        if isinstance(execution_result, dict) and isinstance(execution_result.get("candidate_manifest"), dict)
        else {}
    )
    return str(execution_candidate.get("target_source") or manifest_target_source(manifest) or "")


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(candidate.get("track_id") or ""),
        str(candidate.get("model_id") or ""),
        str(candidate.get("audio_path") or ""),
    )


def minus_route_id_for_candidate(candidate: dict[str, Any], model_id: str) -> str:
    route_family = str(candidate.get("route_family") or "")
    if route_family == "cpu_dsp_minus_phase_repair":
        return f"{model_id}__{route_family}"
    return model_id


def minus_candidate_id(*, track_id: str, route_id: str, role_metadata: dict[str, str]) -> str:
    source_stem_type = str(role_metadata.get("source_manifest_stem_type") or "")
    source_suffix = "__source_instrumental" if source_stem_type == "instrumental" else ""
    return f"{track_id}__minus__{route_id}{source_suffix}"


def build_vocal_companion_map(candidates: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    companions: dict[str, dict[str, Any]] = {}
    for candidate in sorted(candidates, key=candidate_sort_key):
        target_stem = str(candidate.get("target_stem") or "").lower()
        track_id = str(candidate.get("track_id") or "")
        if not track_id:
            continue
        manifest_path = Path(str(candidate.get("candidate_manifest") or ""))
        vocal_entry: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            vocal_entry = explicit_vocal_entry(manifest)
        vocal_path = str(vocal_entry.get("path") or "")
        if not vocal_path and target_stem in VOCAL_TARGET_STEMS:
            vocal_path = str(candidate.get("audio_path") or "")
            vocal_entry = {"path": vocal_path, "source": "target_audio_path", "model_output": True}
        if vocal_path:
            role = vocal_entry_role(target_stem=target_stem, entry=vocal_entry)
            companions.setdefault(
                track_id,
                vocal_entry_payload(
                    path=vocal_path,
                    role=role,
                    source=str(vocal_entry.get("source") or ""),
                    model_id=str(candidate.get("model_id") or ""),
                ),
            )
    return companions


def build_minus_row(
    candidate: dict[str, Any], *, vocal_companions: dict[str, dict[str, Any]] | None = None
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    target_stem = str(candidate.get("target_stem") or "").lower()
    if target_stem not in MINUS_TARGET_STEMS | VOCAL_TARGET_STEMS:
        return None, None

    manifest_path = Path(str(candidate.get("candidate_manifest") or ""))
    if not manifest_path.exists():
        if target_stem in VOCAL_TARGET_STEMS:
            return None, None
        return None, {"track_id": str(candidate.get("track_id") or ""), "skip_reason": "candidate_manifest_missing"}

    manifest = read_json(manifest_path)
    target_source = candidate_target_source(candidate, manifest).lower()
    if target_source in FORBIDDEN_MINUS_TARGET_SOURCES:
        return None, {
            "track_id": str(candidate.get("track_id") or ""),
            "skip_reason": "stem_derived_minus_candidate_rejected",
            "target_source": target_source,
        }
    stem_paths = manifest_stem_paths(manifest)
    minus_path = explicit_minus_path(stem_paths)
    if target_stem in MINUS_TARGET_STEMS:
        minus_path = str(candidate.get("audio_path") or minus_path or manifest.get("target_audio_path") or "")
    elif not minus_path:
        return None, None
    track_id = str(candidate.get("track_id") or "")
    own_vocal_entry = explicit_vocal_entry(manifest)
    own_vocal_path = str(own_vocal_entry.get("path") or explicit_vocal_path(stem_paths))
    if own_vocal_path:
        vocal_info = vocal_entry_payload(
            path=own_vocal_path,
            role=vocal_entry_role(target_stem=target_stem, entry=own_vocal_entry),
            source=str(own_vocal_entry.get("source") or ""),
            model_id=str(candidate.get("model_id") or manifest.get("model_id") or manifest.get("model_name") or ""),
        )
    else:
        vocal_info = dict((vocal_companions or {}).get(track_id, {}))
    vocal_path = str(vocal_info.get("path") or "")
    if not minus_path:
        return None, {"track_id": track_id, "skip_reason": "missing_minus_path"}
    if not vocal_path:
        return None, {"track_id": track_id, "skip_reason": "missing_vocal_companion_path"}
    if minus_path == vocal_path:
        return None, {"track_id": track_id, "skip_reason": "minus_path_matches_vocal_path"}

    model_id = str(candidate.get("model_id") or manifest.get("model_id") or manifest.get("model_name") or "minus")
    route_id = minus_route_id_for_candidate(candidate, model_id)
    source_audio_path = str(
        candidate.get("source_audio_path") or manifest.get("source_audio_path") or manifest.get("audio_path") or ""
    )
    role_metadata = minus_role_metadata(
        candidate=candidate,
        manifest=manifest,
        stem_paths=stem_paths,
        minus_path=minus_path,
    )
    row = {
            "schema": "testing_stuff_music_minus_windowed_judge_candidate_row.v1",
            "track_id": track_id,
            "candidate_id": minus_candidate_id(track_id=track_id, route_id=route_id, role_metadata=role_metadata),
            "mix_path": source_audio_path,
            "source_audio_path": source_audio_path,
            "minus_path": minus_path,
            "instrumental_path": minus_path,
            "selected_minus_path": minus_path,
            "vocal_companion_path": vocal_path,
            "vocal_companion_role": str(vocal_info.get("role") or ""),
            "vocal_companion_source": str(vocal_info.get("source") or ""),
            "vocal_companion_model_id": str(vocal_info.get("model_id") or ""),
            "stem_paths": stem_paths,
            "candidate_manifest": str(manifest_path),
            "source_manifest": str(manifest_path),
            "score_mode": "minus",
            "stem_type": "minus",
            "model_id": model_id,
            "minus_model_id": model_id,
            "minus_route_id": route_id,
            "route_family": str(candidate.get("route_family") or ""),
            "source_execution_results_jsonl": str(candidate.get("source_execution_results_jsonl") or ""),
            **role_metadata,
            "delivery_allowed": False,
            "presentation_status": "diagnostic_candidate_for_minus_judge_only",
            "release_status": "not_release_ready",
            "release_ready": False,
            "forbidden_actions": list(FORBIDDEN_ACTIONS),
        }
    if vocal_info.get("is_selected_vocal_candidate") is True:
        row["vocal_path"] = vocal_path
        row["selected_vocal_path"] = vocal_path
        row["vocal_path_role"] = "selected_vocal_candidate"
    return (row, None)


def build_minus_rows(candidates: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    vocal_companions = build_vocal_companion_map(candidates)
    for candidate in sorted(
        candidates,
        key=candidate_sort_key,
    ):
        row, skip = build_minus_row(candidate, vocal_companions=vocal_companions)
        if skip is not None:
            skipped.append(skip)
        if row is not None:
            rows.append(row)
    return rows, skipped


def build_summary(
    *,
    execution_results_jsonls: Sequence[Path],
    completed_candidate_count: int,
    rows: Sequence[dict[str, Any]],
    skipped: Sequence[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema": "testing_stuff_music_corpus_minus_judge_inputs_summary.v1",
        "source_execution_results_jsonls": [source_ref(path) for path in execution_results_jsonls],
        "completed_candidate_count": completed_candidate_count,
        "candidate_count": len(rows),
        "track_count": len({str(row.get("track_id") or "") for row in rows}),
        "skipped_candidate_count": len(skipped),
        "skipped_candidates": list(skipped),
        "expected_downstream": "run_no_reference_windowed_judge --score-mode minus",
        "expected_acceleration_mode": "external_only_cpu_dsp_scoring",
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "builds_minus_windowed_judge_inputs_from_existing_completed_candidates",
        "delivery_allowed": False,
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "testing_stuff_music_corpus_minus_judge_inputs_manifest.v1",
        "manifest_type": "testing_stuff_music_corpus_minus_judge_inputs",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_candidate_rows_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "candidate_count": len(rows),
        "summary": summary,
        "safety": dict(SAFETY_FLAGS),
    }


def build_summary_md(manifest: dict[str, Any]) -> str:
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    lines = [
        "# Testing Stuff Music Minus Judge Inputs",
        "",
        f"- candidate_count: {summary.get('candidate_count', 0)}",
        f"- track_count: {summary.get('track_count', 0)}",
        f"- skipped_candidate_count: {summary.get('skipped_candidate_count', 0)}",
        f"- expected_downstream: {summary.get('expected_downstream')}",
        f"- expected_acceleration_mode: {summary.get('expected_acceleration_mode')}",
        "- delivery_allowed: false",
        "- release_ready: false",
        "",
    ]
    return "\n".join(lines)


def build_inputs(*, execution_results_jsonls: Sequence[Path]) -> dict[str, Any]:
    candidates = collect_completed_candidates(execution_results_jsonls)
    rows, skipped = build_minus_rows(candidates)
    summary = build_summary(
        execution_results_jsonls=execution_results_jsonls,
        completed_candidate_count=len(candidates),
        rows=rows,
        skipped=skipped,
    )
    return {"rows": rows, "summary": summary, "manifest": build_manifest(rows, summary)}


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    write_jsonl(output_dir / CANDIDATES_JSONL_NAME, result["rows"])
    write_json(output_dir / MANIFEST_JSON_NAME, result["manifest"])
    (output_dir / SUMMARY_MD_NAME).parent.mkdir(parents=True, exist_ok=True)
    (output_dir / SUMMARY_MD_NAME).write_text(build_summary_md(result["manifest"]), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build minus windowed-judge candidate rows from completed testing_stuff_music corpus execution rows."
    )
    parser.add_argument("--execution-results-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_testing_stuff_music_corpus_minus_judge_inputs.py",
        entrypoint_path="/app/tools/build_testing_stuff_music_corpus_minus_judge_inputs.py",
    )
    args = parse_args(argv)
    result = build_inputs(execution_results_jsonls=args.execution_results_jsonl)
    write_outputs(result, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
