from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


RECOVERY_STATUSES = {
    "reference_found",
    "reference_candidate_found",
    "blocked_missing_reference",
    "blocked_ambiguous_reference",
    "blocked_reference_outside_allowed_roots",
}
DEFAULT_REFERENCE_SEARCH_ROOTS = [
    Path("/app/lab_artifacts"),
    Path("/host_testing_stuff_music"),
    Path("/app/testing_flstudio_music"),
]
AUDIO_SUFFIXES = (".flac", ".wav", ".mp3", ".aiff", ".aif", ".ogg", ".m4a")
REFERENCE_KEY_TOKENS = {
    "vocal": (
        "reference_vocal",
        "vocal_reference",
        "ground_truth_vocal",
        "gt_vocal",
        "target_reference_vocal",
    ),
    "minus": (
        "reference_minus",
        "minus_reference",
        "reference_instrumental",
        "instrumental_reference",
        "ground_truth_minus",
        "ground_truth_instrumental",
        "gt_minus",
        "target_reference_minus",
    ),
}
REFERENCE_NAME_TOKENS = {
    "vocal": ("reference", "ground_truth", "gt", "truth"),
    "minus": ("reference", "ground_truth", "gt", "truth"),
}
STEM_NAME_TOKENS = {
    "vocal": ("vocal", "vocals"),
    "minus": ("minus", "instrumental"),
}
GENERATED_OR_PROXY_TOKENS = (
    "candidate",
    "selected",
    "listen",
    "generated",
    "adaptive",
    "ir_residual",
    "instrumental_resurrection",
    "proxy",
)
SAFETY_FLAGS = {
    "audio_copied": False,
    "audio_read": False,
    "audio_hashed": False,
    "gt_bundle_built": False,
    "metrics_run": False,
    "training_started": False,
    "policy_changed": False,
    "models_changed": False,
    "weights_added": False,
    "cleanup_routes_added": False,
    "thresholds_changed": False,
}
DEFAULT_FORBIDDEN_ACTIONS = [
    "no_audio_copy",
    "no_audio_export_as_result",
    "no_audio_read_or_hash",
    "no_gt_bundle_build_in_phase10f",
    "no_metrics_run_in_phase10f",
    "no_promotion",
    "no_training_start",
    "no_policy_change",
    "no_threshold_change",
    "no_cleanup_debleed_denoise",
    "no_model_or_weight_addition",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"json_must_be_object:{path}")
    return payload


def try_read_json(path: Path | None) -> tuple[dict[str, Any] | None, str]:
    if path is None:
        return None, "source_manifest_missing"
    if not path.exists():
        return None, "source_manifest_not_found"
    try:
        return read_json(path), ""
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, f"source_manifest_unreadable:{type(exc).__name__}"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"jsonl_row_must_be_object:{path}:{line_number}")
            yield payload


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]
    text = str(value).strip()
    return [text] if text else []


def walk_values(payload: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            current = prefix + (str(key),)
            yield current, value
            yield from walk_values(value, current)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            current = prefix + (str(index),)
            yield current, value
            yield from walk_values(value, current)


def load_golden_track_hashes(golden_jsonl: Path) -> set[str]:
    return {str(row.get("track_hash") or "").strip() for row in iter_jsonl(golden_jsonl) if row.get("track_hash")}


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or row.get("benchmark_case_id") or "")


def rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {row_id(row): row for row in iter_jsonl(path) if row_id(row)}


def resolve_manifest_path(manifest_path: Path | None, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() or manifest_path is None:
        return candidate
    return (manifest_path.parent / candidate).resolve()


def safe_resolve(path: Path) -> Path:
    return path.resolve(strict=False)


def path_under_allowed_root(path: Path, allowed_roots: list[Path]) -> bool:
    resolved_path = safe_resolve(path)
    for root in allowed_roots:
        resolved_root = safe_resolve(root)
        if resolved_path == resolved_root or resolved_root in resolved_path.parents:
            return True
    return False


def is_audio_path(text: str) -> bool:
    return text.lower().endswith(AUDIO_SUFFIXES)


def has_path_separator(text: str) -> bool:
    return "/" in text or "\\" in text


def target_reference_type(stem_type: str) -> str:
    return "reference_minus" if stem_type == "minus" else "reference_vocal"


def missing_reference_type(stem_type: str) -> str:
    return target_reference_type(stem_type)


def is_generated_or_proxy_path(key: str, value: str) -> bool:
    key_l = key.lower()
    if any(token in key_l for token in GENERATED_OR_PROXY_TOKENS):
        return True
    value_l = value.lower().replace("\\", "/")
    basename = Path(value_l).name
    path_blockers = (
        "/candidates/",
        "/selected",
        "/listen_",
        "/adaptive_",
        "/ir_residual",
        "/instrumental_resurrection",
        "/generated",
        "selected_",
        "generated_",
        "adaptive_",
    )
    return any(token in value_l or basename.startswith(token) for token in path_blockers)


def is_reference_candidate_key(key: str, value: str, stem_type: str) -> bool:
    key_l = key.lower()
    value_l = value.lower()
    if not (has_path_separator(value) and is_audio_path(value)):
        return False
    if is_generated_or_proxy_path(key_l, value_l):
        return False
    stem_key_tokens = REFERENCE_KEY_TOKENS.get(stem_type, REFERENCE_KEY_TOKENS["vocal"])
    if any(token in key_l for token in stem_key_tokens):
        return True
    if "reference" in key_l or "ground_truth" in key_l or "gt_" in key_l:
        return any(token in value_l for token in STEM_NAME_TOKENS.get(stem_type, (stem_type,)))
    return False


def is_reference_candidate_name(path: Path, stem_type: str) -> bool:
    lower = str(path).lower()
    if not is_audio_path(lower) or is_generated_or_proxy_path("", lower):
        return False
    return any(token in lower for token in REFERENCE_NAME_TOKENS.get(stem_type, ())) and any(
        token in lower for token in STEM_NAME_TOKENS.get(stem_type, (stem_type,))
    )


def make_candidate_record(
    *,
    input_type: str,
    path: Path,
    source_key: str,
    method: str,
    allowed_roots: list[Path],
) -> dict[str, Any]:
    return {
        "type": input_type,
        "path": str(path),
        "exists": path.exists(),
        "allowed_root": path_under_allowed_root(path, allowed_roots),
        "source_key": source_key,
        "candidate_match_method": method,
    }


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (str(item["type"]), str(item["path"]), str(item["source_key"]), str(item["candidate_match_method"])),
    ):
        key = (str(record["type"]), str(record["path"]), str(record["candidate_match_method"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def discover_source_audio_paths(mapping_row: dict[str, Any], reference_recovery_row: dict[str, Any] | None) -> list[str]:
    source_paths: set[str] = set()
    for record in mapping_row.get("original_paths_found") or []:
        if not isinstance(record, dict):
            continue
        if record.get("type") in {"original_mix", "full_mix_reference"} and record.get("exists_in_docker") is True:
            source_paths.add(str(record.get("path") or ""))
    for record in (reference_recovery_row or {}).get("discovered_audio_inputs") or []:
        if not isinstance(record, dict):
            continue
        if record.get("type") in {"original_mix", "full_mix_reference"} and record.get("exists") is True:
            source_paths.add(str(record.get("path") or ""))
    return sorted(path for path in source_paths if path)


def has_flstudio_export(mapping_row: dict[str, Any], reference_recovery_row: dict[str, Any] | None) -> bool:
    records = list(mapping_row.get("original_paths_found") or []) + list(
        (reference_recovery_row or {}).get("discovered_audio_inputs") or []
    )
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("type") == "flstudio_stem_export" and (record.get("exists") is True or record.get("exists_in_docker") is True):
            return True
    return False


def discover_manifest_reference_candidates(
    *,
    source_manifest_path: Path | None,
    source_payload: dict[str, Any] | None,
    stem_type: str,
    allowed_roots: list[Path],
) -> list[dict[str, Any]]:
    if source_payload is None:
        return []
    candidates: list[dict[str, Any]] = []
    input_type = target_reference_type(stem_type)
    for path_parts, value in walk_values(source_payload):
        if not isinstance(value, str) or not value.strip():
            continue
        source_key = ".".join(path_parts)
        path_text = value.strip()
        if not is_reference_candidate_key(source_key, path_text, stem_type):
            continue
        candidates.append(
            make_candidate_record(
                input_type=input_type,
                path=resolve_manifest_path(source_manifest_path, path_text),
                source_key=source_key,
                method="manifest_reference_path",
                allowed_roots=allowed_roots,
            )
        )
    return candidates


def discover_existing_reference_records(
    *,
    reference_recovery_row: dict[str, Any] | None,
    stem_type: str,
    allowed_roots: list[Path],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    input_type = target_reference_type(stem_type)
    for record in (reference_recovery_row or {}).get("discovered_reference_inputs") or []:
        if not isinstance(record, dict) or record.get("type") != input_type:
            continue
        raw_path = str(record.get("path") or "")
        if not raw_path:
            continue
        candidates.append(
            make_candidate_record(
                input_type=input_type,
                path=Path(raw_path),
                source_key=str(record.get("source_key") or "discovered_reference_inputs"),
                method="existing_reference_record",
                allowed_roots=allowed_roots,
            )
        )
    return candidates


def discover_filesystem_reference_candidates(
    *,
    reference_search_roots: list[Path],
    stem_type: str,
    allowed_roots: list[Path],
) -> list[dict[str, Any]]:
    # Conservative search: only files explicitly named as reference/ground-truth for the target stem.
    # Generated candidate/listen artifacts are deliberately filtered out.
    candidates: list[dict[str, Any]] = []
    for root in sorted(reference_search_roots, key=lambda item: str(item)):
        if not root.exists() or not root.is_dir():
            continue
        for path in sorted(root.rglob("*"), key=lambda item: str(item)):
            if not path.is_file() or not is_reference_candidate_name(path, stem_type):
                continue
            candidates.append(
                make_candidate_record(
                    input_type=target_reference_type(stem_type),
                    path=path,
                    source_key=f"filesystem_search:{root}",
                    method="filesystem_reference_search",
                    allowed_roots=allowed_roots,
                )
            )
    return candidates


def candidate_methods(candidates: list[dict[str, Any]]) -> str:
    methods = sorted({str(candidate.get("candidate_match_method") or "") for candidate in candidates if candidate.get("candidate_match_method")})
    return "+".join(methods) if methods else "none"


def decide_recovery_status(candidates: list[dict[str, Any]]) -> tuple[str, str, list[dict[str, Any]]]:
    existing = [candidate for candidate in candidates if candidate.get("exists") is True]
    allowed = [candidate for candidate in existing if candidate.get("allowed_root") is True]
    outside = [candidate for candidate in existing if candidate.get("allowed_root") is False]
    if len(allowed) == 1:
        return "reference_candidate_found", "", allowed
    if len(allowed) > 1:
        return "blocked_ambiguous_reference", "multiple_allowed_reference_candidates", allowed
    if outside:
        return "blocked_reference_outside_allowed_roots", "reference_candidate_outside_allowed_roots", outside
    return "blocked_missing_reference", "no_isolated_reference_candidate_found", []


def required_next_tool_for(status: str) -> str:
    if status in {"reference_found", "reference_candidate_found"}:
        return "run_ground_truth_quality_gate"
    if status == "blocked_ambiguous_reference":
        return "manual_source_manifest_inspection"
    if status == "blocked_reference_outside_allowed_roots":
        return "manual_source_manifest_inspection"
    return "manual_source_manifest_inspection"


def recovery_row(
    mapping_row: dict[str, Any],
    *,
    reference_recovery_row: dict[str, Any] | None,
    golden_hashes: set[str],
    reference_search_roots: list[Path],
    enable_filesystem_search: bool,
) -> dict[str, Any]:
    stem_type = str(mapping_row.get("stem_type") or "unknown")
    source_manifest = str(mapping_row.get("source_manifest") or "")
    source_manifest_path = Path(source_manifest) if source_manifest else None
    source_payload, source_manifest_error = try_read_json(source_manifest_path)
    allowed_roots = reference_search_roots
    explicit_candidates = dedupe_records(
        discover_existing_reference_records(
            reference_recovery_row=reference_recovery_row,
            stem_type=stem_type,
            allowed_roots=allowed_roots,
        )
        + discover_manifest_reference_candidates(
            source_manifest_path=source_manifest_path,
            source_payload=source_payload,
            stem_type=stem_type,
            allowed_roots=allowed_roots,
        )
    )
    filesystem_candidates = (
        []
        if explicit_candidates or not enable_filesystem_search
        else discover_filesystem_reference_candidates(
            reference_search_roots=reference_search_roots,
            stem_type=stem_type,
            allowed_roots=allowed_roots,
        )
    )
    candidates = dedupe_records(explicit_candidates + filesystem_candidates)
    status, blocked_reason, decisive_candidates = decide_recovery_status(candidates)
    if source_manifest_error and status == "blocked_missing_reference":
        blocked_reason = source_manifest_error
    if status not in RECOVERY_STATUSES:
        raise ValueError(f"invalid_reference_path_recovery_status:{status}")

    source_audio_paths = discover_source_audio_paths(mapping_row, reference_recovery_row)
    can_recover_reference = status in {"reference_found", "reference_candidate_found"}
    can_run_metrics_after = can_recover_reference and bool(source_audio_paths)
    can_build_gt_bundle_after = bool(source_audio_paths) and has_flstudio_export(mapping_row, reference_recovery_row)
    forbidden_actions = sorted(
        set(
            as_list(mapping_row.get("forbidden_actions"))
            + as_list((reference_recovery_row or {}).get("forbidden_actions"))
            + DEFAULT_FORBIDDEN_ACTIONS
        )
    )
    if str(mapping_row.get("split") or "") == "holdout":
        forbidden_actions.append("holdout_protected")
    if str(mapping_row.get("track_hash") or "") in golden_hashes:
        forbidden_actions.append("golden_protected_no_training_or_tuning")

    missing_reference_types = [] if can_recover_reference else [missing_reference_type(stem_type)]
    if source_manifest_error and "missing_source_manifest" not in missing_reference_types:
        missing_reference_types.append("missing_source_manifest")
    return {
        "case_id": str(mapping_row.get("case_id") or ""),
        "track_hash": str(mapping_row.get("track_hash") or ""),
        "stem_type": stem_type,
        "split": str(mapping_row.get("split") or ""),
        "source_manifest": source_manifest,
        "current_mapping_status": str(mapping_row.get("mapping_status") or ""),
        "discovered_source_audio_paths": source_audio_paths,
        "discovered_reference_candidates": candidates,
        "missing_reference_types": missing_reference_types,
        "reference_search_roots": [str(root) for root in reference_search_roots],
        "candidate_match_method": candidate_methods(decisive_candidates or candidates),
        "recovery_status": status,
        "can_recover_reference_audio": can_recover_reference,
        "can_run_metrics_after_reference_recovery": can_run_metrics_after,
        "can_build_gt_bundle_after_reference_recovery": can_build_gt_bundle_after,
        "required_next_tool": required_next_tool_for(status),
        "blocked_reason": blocked_reason,
        "forbidden_actions": sorted(set(forbidden_actions)),
    }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "") for row in rows).most_common())


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "reference_path_recovery_summary.v1",
        "total_rows": len(rows),
        "recovery_status_counts": count_by(rows, "recovery_status"),
        "required_next_tool_counts": count_by(rows, "required_next_tool"),
        "stem_type_counts": count_by(rows, "stem_type"),
        "reference_found_count": sum(1 for row in rows if row["recovery_status"] == "reference_found"),
        "reference_candidate_found_count": sum(1 for row in rows if row["recovery_status"] == "reference_candidate_found"),
        "blocked_missing_reference_count": sum(1 for row in rows if row["recovery_status"] == "blocked_missing_reference"),
        "blocked_ambiguous_reference_count": sum(
            1 for row in rows if row["recovery_status"] == "blocked_ambiguous_reference"
        ),
        "blocked_reference_outside_allowed_roots_count": sum(
            1 for row in rows if row["recovery_status"] == "blocked_reference_outside_allowed_roots"
        ),
        "can_recover_reference_audio_count": sum(1 for row in rows if row["can_recover_reference_audio"]),
        "can_run_metrics_after_reference_recovery_count": sum(
            1 for row in rows if row["can_run_metrics_after_reference_recovery"]
        ),
        "can_build_gt_bundle_after_reference_recovery_count": sum(
            1 for row in rows if row["can_build_gt_bundle_after_reference_recovery"]
        ),
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(
    *,
    summary: dict[str, Any],
    source_path_mapping_jsonl: Path,
    source_path_mapping_manifest_json: Path,
    docker_mount_verification_json: Path,
    reference_recovery_plan_jsonl: Path,
    collected_evidence_jsonl: Path,
    hard_case_benchmark_jsonl: Path,
    golden_jsonl: Path,
) -> dict[str, Any]:
    input_files = {
        "source_path_mapping_jsonl": str(source_path_mapping_jsonl),
        "source_path_mapping_manifest_json": str(source_path_mapping_manifest_json),
        "docker_mount_verification_json": str(docker_mount_verification_json),
        "reference_recovery_plan_jsonl": str(reference_recovery_plan_jsonl),
        "collected_evidence_jsonl": str(collected_evidence_jsonl),
        "hard_case_benchmark_jsonl": str(hard_case_benchmark_jsonl),
        "golden_jsonl": str(golden_jsonl),
    }
    return {
        "schema": "reference_path_recovery_manifest.v1",
        **summary,
        "input_files": input_files,
        "input_sha256": {
            key: sha256_file(Path(value))
            for key, value in input_files.items()
            if Path(value).exists()
        },
        "safety": dict(SAFETY_FLAGS),
    }


def build_reference_path_recovery_plan(
    *,
    source_path_mapping_jsonl: Path,
    source_path_mapping_manifest_json: Path,
    docker_mount_verification_json: Path,
    reference_recovery_plan_jsonl: Path,
    collected_evidence_jsonl: Path,
    hard_case_benchmark_jsonl: Path,
    golden_jsonl: Path,
    reference_search_roots: list[Path] | None = None,
) -> dict[str, Any]:
    _source_mapping_manifest = read_json(source_path_mapping_manifest_json)
    _docker_verification = read_json(docker_mount_verification_json)
    reference_rows = rows_by_id(reference_recovery_plan_jsonl)
    _collected_rows = rows_by_id(collected_evidence_jsonl)
    _benchmark_rows = rows_by_id(hard_case_benchmark_jsonl)
    golden_hashes = load_golden_track_hashes(golden_jsonl)
    roots = [Path(root) for root in (reference_search_roots or DEFAULT_REFERENCE_SEARCH_ROOTS)]
    enable_filesystem_search = reference_search_roots is not None
    rows = [
        recovery_row(
            row,
            reference_recovery_row=reference_rows.get(str(row.get("case_id") or "")),
            golden_hashes=golden_hashes,
            reference_search_roots=roots,
            enable_filesystem_search=enable_filesystem_search,
        )
        for row in iter_jsonl(source_path_mapping_jsonl)
        if row.get("mapping_status") == "blocked_missing_reference_path"
    ]
    rows.sort(key=lambda row: row["case_id"])
    summary = build_summary(rows)
    manifest = build_manifest(
        summary=summary,
        source_path_mapping_jsonl=source_path_mapping_jsonl,
        source_path_mapping_manifest_json=source_path_mapping_manifest_json,
        docker_mount_verification_json=docker_mount_verification_json,
        reference_recovery_plan_jsonl=reference_recovery_plan_jsonl,
        collected_evidence_jsonl=collected_evidence_jsonl,
        hard_case_benchmark_jsonl=hard_case_benchmark_jsonl,
        golden_jsonl=golden_jsonl,
    )
    return {"rows": rows, "summary": summary, "manifest": manifest}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Reference Path Recovery Summary",
        "",
        f"- `total_rows`: {summary['total_rows']}",
        f"- `reference_found_count`: {summary['reference_found_count']}",
        f"- `reference_candidate_found_count`: {summary['reference_candidate_found_count']}",
        f"- `blocked_missing_reference_count`: {summary['blocked_missing_reference_count']}",
        f"- `blocked_ambiguous_reference_count`: {summary['blocked_ambiguous_reference_count']}",
        f"- `blocked_reference_outside_allowed_roots_count`: {summary['blocked_reference_outside_allowed_roots_count']}",
        f"- `can_recover_reference_audio_count`: {summary['can_recover_reference_audio_count']}",
        f"- `can_run_metrics_after_reference_recovery_count`: {summary['can_run_metrics_after_reference_recovery_count']}",
        f"- `can_build_gt_bundle_after_reference_recovery_count`: {summary['can_build_gt_bundle_after_reference_recovery_count']}",
        "",
        "## Recovery Status",
        "",
    ]
    for key, value in summary["recovery_status_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Required Next Tool", ""])
    for key, value in summary["required_next_tool_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "reference_path_recovery_plan.jsonl", result["rows"])
    (output_dir / "reference_path_recovery_manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary_md(output_dir / "reference_path_recovery_summary.md", result["summary"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan reference path recovery for remaining blocked reference rows.")
    parser.add_argument("--source-path-mapping-jsonl", type=Path, required=True)
    parser.add_argument("--source-path-mapping-manifest-json", type=Path, required=True)
    parser.add_argument("--docker-mount-verification-json", type=Path, required=True)
    parser.add_argument("--reference-recovery-plan-jsonl", type=Path, required=True)
    parser.add_argument("--collected-evidence-jsonl", type=Path, required=True)
    parser.add_argument("--hard-case-benchmark-jsonl", type=Path, required=True)
    parser.add_argument("--golden-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-search-root", type=Path, action="append", default=None)
    args = parser.parse_args()
    result = build_reference_path_recovery_plan(
        source_path_mapping_jsonl=args.source_path_mapping_jsonl,
        source_path_mapping_manifest_json=args.source_path_mapping_manifest_json,
        docker_mount_verification_json=args.docker_mount_verification_json,
        reference_recovery_plan_jsonl=args.reference_recovery_plan_jsonl,
        collected_evidence_jsonl=args.collected_evidence_jsonl,
        hard_case_benchmark_jsonl=args.hard_case_benchmark_jsonl,
        golden_jsonl=args.golden_jsonl,
        reference_search_roots=args.reference_search_root,
    )
    write_outputs(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
