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


MAPPING_STATUSES = {
    "mapping_found",
    "mapping_candidate_found",
    "blocked_missing_host_path",
    "blocked_missing_docker_mount",
    "blocked_missing_reference_path",
    "blocked_ambiguous_mapping",
}
AUDIO_TYPES = {
    "original_mix",
    "generated_vocal",
    "generated_minus",
    "reference_vocal",
    "reference_minus",
    "full_mix_reference",
    "flstudio_stem_export",
}
REFERENCE_TYPES = {"reference_vocal", "reference_minus", "full_mix_reference"}
SOURCE_TYPES = {"original_mix", "full_mix_reference"}
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
    "no_audio_read",
    "no_audio_hash",
    "no_gt_bundle_build",
    "no_reference_metrics",
    "no_promotion",
    "no_training_start",
    "no_policy_change",
    "no_threshold_change",
    "no_cleanup_debleed_denoise",
    "no_model_or_weight_addition",
]
DEFAULT_PREFIX_MAPPINGS = {
    "/host_testing_stuff_music": "/app/testing_stuff_music",
    "/host_testing_flstudio_music": "/app/testing_flstudio_music",
    "/host_downloads": "/app/host_downloads",
}
DEFAULT_HOST_MOUNT_HINTS = {
    "/host_testing_stuff_music": "./testing_stuff_music:/host_testing_stuff_music:ro",
    "/host_testing_flstudio_music": "./testing_flstudio_music:/host_testing_flstudio_music:ro",
    "/host_downloads": "C:/Users/PBAX/Downloads:/host_downloads:ro",
}
DEFAULT_DOCKER_SEARCH_ROOTS: tuple[Path, ...] = ()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"json_must_be_object:{path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"jsonl_row_must_be_object:{path}:{line_number}")
            yield payload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]
    text = str(value).strip()
    return [text] if text else []


def load_golden_track_hashes(path: Path) -> set[str]:
    return {str(row.get("track_hash") or "").strip() for row in iter_jsonl(path) if row.get("track_hash")}


def normalize_path_text(path: str) -> str:
    return str(path or "").strip().replace("\\", "/")


def path_prefix(path_text: str) -> str:
    normalized = normalize_path_text(path_text)
    if not normalized:
        return ""
    if len(normalized) >= 3 and normalized[1:3] == ":/":
        parts = normalized.split("/")
        return "/".join(parts[:3]) if len(parts) >= 3 else normalized
    if normalized.startswith("/"):
        parts = normalized.split("/")
        if len(parts) >= 3 and parts[1] == "app":
            return "/".join(parts[:3])
        return "/" + normalized.split("/")[1]
    return "."


def docker_path_exists(path_text: str) -> bool:
    return bool(path_text) and Path(path_text).exists()


def all_input_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ("discovered_audio_inputs", "discovered_reference_inputs", "discovered_manifest_inputs"):
        for item in row.get(key) or []:
            if isinstance(item, dict) and item.get("path"):
                records.append(dict(item))
    return records


def original_paths_found(row: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for item in all_input_records(row):
        path_text = normalize_path_text(str(item.get("path") or ""))
        if not path_text:
            continue
        records.append(
            {
                "type": str(item.get("type") or "unknown"),
                "path": path_text,
                "source_key": str(item.get("source_key") or ""),
                "exists_in_docker": docker_path_exists(path_text),
            }
        )
    records.sort(key=lambda value: (value["type"], value["path"], value["source_key"]))
    return dedupe_dicts(records, ("type", "path", "source_key"))


def dedupe_dicts(records: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    output: list[dict[str, Any]] = []
    for record in records:
        key = tuple(record.get(item) for item in keys)
        if key in seen:
            continue
        seen.add(key)
        output.append(record)
    return output


def path_basename(path_text: str) -> str:
    return Path(normalize_path_text(path_text)).name


def infer_host_path_candidates(path_text: str) -> list[dict[str, Any]]:
    normalized = normalize_path_text(path_text)
    candidates: list[dict[str, Any]] = []
    if normalized.startswith("/host_testing_stuff_music/"):
        rel = normalized.removeprefix("/host_testing_stuff_music/")
        candidates.append(
            {
                "path": f"./testing_stuff_music/{rel}",
                "mapping_prefix": "/host_testing_stuff_music",
                "reason": "legacy_host_testing_stuff_music_prefix",
            }
        )
    elif normalized.startswith("/host_testing_flstudio_music/"):
        rel = normalized.removeprefix("/host_testing_flstudio_music/")
        candidates.append(
            {
                "path": f"./testing_flstudio_music/{rel}",
                "mapping_prefix": "/host_testing_flstudio_music",
                "reason": "legacy_host_testing_flstudio_music_prefix",
            }
        )
    elif normalized.startswith("/host_downloads/"):
        rel = normalized.removeprefix("/host_downloads/")
        candidates.append(
            {
                "path": f"C:/Users/PBAX/Downloads/{rel}",
                "mapping_prefix": "/host_downloads",
                "reason": "documented_host_downloads_prefix",
            }
        )
    return candidates


def configured_candidate(path_text: str, configured_prefix_mappings: dict[str, str]) -> str | None:
    normalized = normalize_path_text(path_text)
    for source_prefix, target_prefix in sorted(configured_prefix_mappings.items(), key=lambda item: len(item[0]), reverse=True):
        source = normalize_path_text(source_prefix).rstrip("/")
        target = normalize_path_text(target_prefix).rstrip("/")
        if normalized == source:
            return target
        if normalized.startswith(f"{source}/"):
            suffix = normalized[len(source) + 1 :]
            return f"{target}/{suffix}"
    return None


def find_basename_candidates(path_text: str, docker_search_roots: list[Path]) -> list[str]:
    name = path_basename(path_text)
    if not name:
        return []
    candidates: list[str] = []
    for root in sorted(docker_search_roots, key=lambda item: str(item)):
        if not root.exists() or not root.is_dir():
            continue
        for candidate in sorted(root.rglob(name), key=lambda item: str(item)):
            if candidate.is_file():
                candidates.append(str(candidate))
    return sorted(set(candidates))


def docker_visible_candidates(
    path_text: str,
    *,
    configured_prefix_mappings: dict[str, str],
    docker_search_roots: list[Path],
) -> list[str]:
    candidates: list[str] = []
    direct = configured_candidate(path_text, configured_prefix_mappings)
    if direct and Path(direct).exists():
        candidates.append(direct)
    candidates.extend(find_basename_candidates(path_text, docker_search_roots))
    return sorted(set(candidates))


def missing_source_paths(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item for item in paths
        if item["type"] in SOURCE_TYPES and not item["exists_in_docker"]
    ]


def source_paths(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in paths if item["type"] in SOURCE_TYPES]


def reference_paths(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in paths if item["type"] in REFERENCE_TYPES]


def reference_available(paths: list[dict[str, Any]], candidates_by_missing_path: dict[str, list[str]]) -> bool:
    refs = reference_paths(paths)
    if not refs:
        return False
    for item in refs:
        if item["exists_in_docker"] or candidates_by_missing_path.get(item["path"]):
            return True
    return False


def source_available(paths: list[dict[str, Any]], candidates_by_missing_path: dict[str, list[str]]) -> bool:
    sources = source_paths(paths)
    if not sources:
        return False
    for item in sources:
        if item["exists_in_docker"] or candidates_by_missing_path.get(item["path"]):
            return True
    return False


def mapping_rule_for_prefix(prefixes: list[str]) -> dict[str, Any] | None:
    for prefix in prefixes:
        if prefix in DEFAULT_HOST_MOUNT_HINTS:
            return {
                "source_prefix": prefix,
                "compose_volume": DEFAULT_HOST_MOUNT_HINTS[prefix],
                "reason": "docker_prefix_absent_or_not_mounted",
            }
    return None


def decide_mapping_status(
    *,
    row: dict[str, Any],
    paths: list[dict[str, Any]],
    missing_paths: list[dict[str, Any]],
    candidate_paths: list[str],
    host_only_prefixes: list[str],
    can_recover_source_audio: bool,
    can_recover_reference_audio: bool,
) -> str:
    if len(candidate_paths) > 1:
        return "blocked_ambiguous_mapping"
    if row.get("recovery_status") == "blocked_missing_reference" and not can_recover_reference_audio:
        return "blocked_missing_reference_path"
    if len(candidate_paths) == 1:
        return "mapping_candidate_found"
    if not missing_paths:
        return "mapping_found"
    if host_only_prefixes:
        return "blocked_missing_docker_mount"
    if not can_recover_source_audio:
        return "blocked_missing_host_path"
    return "blocked_missing_reference_path"


def path_mapping_row(
    row: dict[str, Any],
    *,
    golden_track_hashes: set[str],
    configured_prefix_mappings: dict[str, str],
    docker_search_roots: list[Path],
) -> dict[str, Any]:
    paths = original_paths_found(row)
    missing = [item for item in paths if item["type"] in AUDIO_TYPES and not item["exists_in_docker"]]
    prefixes = sorted({path_prefix(item["path"]) for item in paths if item.get("path")})
    host_only_prefixes = [prefix for prefix in prefixes if prefix.startswith("/host_") and not Path(prefix).exists()]

    candidates_by_missing: dict[str, list[str]] = {}
    host_candidates: list[dict[str, Any]] = []
    docker_candidates: list[str] = []
    for item in missing:
        item_candidates = docker_visible_candidates(
            item["path"],
            configured_prefix_mappings=configured_prefix_mappings,
            docker_search_roots=docker_search_roots,
        )
        candidates_by_missing[item["path"]] = item_candidates
        docker_candidates.extend(item_candidates)
        host_candidates.extend(infer_host_path_candidates(item["path"]))

    docker_candidates = sorted(set(docker_candidates))
    host_candidates = dedupe_dicts(
        sorted(host_candidates, key=lambda item: (item["mapping_prefix"], item["path"])),
        ("mapping_prefix", "path"),
    )
    can_recover_source = source_available(paths, candidates_by_missing)
    can_recover_reference = reference_available(paths, candidates_by_missing)
    status = decide_mapping_status(
        row=row,
        paths=paths,
        missing_paths=missing,
        candidate_paths=docker_candidates,
        host_only_prefixes=host_only_prefixes,
        can_recover_source_audio=can_recover_source,
        can_recover_reference_audio=can_recover_reference,
    )
    if status not in MAPPING_STATUSES:
        raise ValueError(f"invalid_mapping_status:{status}")
    suggested_rule = mapping_rule_for_prefix(host_only_prefixes)
    required_docker_mount = str(suggested_rule["compose_volume"]) if suggested_rule else ""
    can_repair = status == "mapping_candidate_found" and len(docker_candidates) == 1
    can_run_phase10e = can_recover_source and can_recover_reference and status in {"mapping_found", "mapping_candidate_found"}
    forbidden_actions = sorted(set(as_list(row.get("forbidden_actions")) + DEFAULT_FORBIDDEN_ACTIONS))
    if str(row.get("split") or "") == "holdout":
        forbidden_actions.append("holdout_protected")
    if str(row.get("track_hash") or "") in golden_track_hashes:
        forbidden_actions.append("golden_protected_no_training_or_tuning")
    return {
        "case_id": str(row.get("case_id") or ""),
        "track_hash": str(row.get("track_hash") or ""),
        "stem_type": str(row.get("stem_type") or "unknown"),
        "split": str(row.get("split") or ""),
        "source_manifest": str(row.get("source_manifest") or ""),
        "original_paths_found": paths,
        "missing_paths": missing,
        "path_prefixes_detected": prefixes,
        "docker_visible_candidate_paths": docker_candidates,
        "host_path_candidates": host_candidates,
        "suggested_mapping_rule": suggested_rule or {},
        "mapping_status": status,
        "can_repair_manifest_paths": can_repair,
        "can_recover_source_audio": can_recover_source,
        "can_recover_reference_audio": can_recover_reference,
        "can_run_phase10e_after_mapping": can_run_phase10e,
        "required_docker_mount": required_docker_mount,
        "forbidden_actions": sorted(set(forbidden_actions)),
    }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "") for row in rows).most_common())


def count_prefixes(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(str(prefix) for prefix in row.get("path_prefixes_detected") or [])
    return dict(counter.most_common())


def count_required_mounts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = str(row.get("required_docker_mount") or "")
        if value:
            counter[value] += 1
    return dict(counter.most_common())


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "source_path_mapping_summary.v1",
        "total_rows": len(rows),
        "mapping_status_counts": count_by(rows, "mapping_status"),
        "detected_path_prefix_counts": count_prefixes(rows),
        "required_docker_mount_counts": count_required_mounts(rows),
        "rows_needing_docker_mount_count": sum(1 for row in rows if row.get("required_docker_mount")),
        "rows_with_missing_reference_path_count": sum(
            1 for row in rows if row.get("mapping_status") == "blocked_missing_reference_path"
        ),
        "rows_with_ambiguous_mapping_count": sum(
            1 for row in rows if row.get("mapping_status") == "blocked_ambiguous_mapping"
        ),
        "can_repair_manifest_paths_count": sum(1 for row in rows if row.get("can_repair_manifest_paths")),
        "can_recover_source_audio_count": sum(1 for row in rows if row.get("can_recover_source_audio")),
        "can_recover_reference_audio_count": sum(1 for row in rows if row.get("can_recover_reference_audio")),
        "can_run_phase10e_after_mapping_count": sum(1 for row in rows if row.get("can_run_phase10e_after_mapping")),
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(
    *,
    summary: dict[str, Any],
    reference_recovery_plan_jsonl: Path,
    reference_recovery_manifest_json: Path,
    collected_evidence_jsonl: Path,
    missing_evidence_plan_jsonl: Path,
    hard_case_benchmark_jsonl: Path,
    golden_jsonl: Path,
) -> dict[str, Any]:
    input_files = {
        "reference_recovery_plan_jsonl": str(reference_recovery_plan_jsonl),
        "reference_recovery_manifest_json": str(reference_recovery_manifest_json),
        "collected_evidence_jsonl": str(collected_evidence_jsonl),
        "missing_evidence_plan_jsonl": str(missing_evidence_plan_jsonl),
        "hard_case_benchmark_jsonl": str(hard_case_benchmark_jsonl),
        "golden_jsonl": str(golden_jsonl),
    }
    return {
        "schema": "source_path_mapping_manifest.v1",
        **summary,
        "input_files": input_files,
        "input_sha256": {
            key: sha256_file(Path(value))
            for key, value in input_files.items()
            if Path(value).exists()
        },
        "safety": dict(SAFETY_FLAGS),
    }


def build_source_path_mapping_plan(
    *,
    reference_recovery_plan_jsonl: Path,
    reference_recovery_manifest_json: Path,
    collected_evidence_jsonl: Path,
    missing_evidence_plan_jsonl: Path,
    hard_case_benchmark_jsonl: Path,
    golden_jsonl: Path,
    configured_prefix_mappings: dict[str, str] | None = None,
    docker_search_roots: list[Path] | None = None,
) -> dict[str, Any]:
    _reference_recovery_manifest = read_json(reference_recovery_manifest_json)
    _collected_rows = list(iter_jsonl(collected_evidence_jsonl))
    _missing_rows = list(iter_jsonl(missing_evidence_plan_jsonl))
    _benchmark_rows = list(iter_jsonl(hard_case_benchmark_jsonl))
    golden_hashes = load_golden_track_hashes(golden_jsonl)
    prefix_mappings = dict(DEFAULT_PREFIX_MAPPINGS)
    prefix_mappings.update(configured_prefix_mappings or {})
    search_roots = list(docker_search_roots) if docker_search_roots is not None else list(DEFAULT_DOCKER_SEARCH_ROOTS)
    rows = [
        path_mapping_row(
            row,
            golden_track_hashes=golden_hashes,
            configured_prefix_mappings=prefix_mappings,
            docker_search_roots=search_roots,
        )
        for row in iter_jsonl(reference_recovery_plan_jsonl)
    ]
    rows.sort(key=lambda item: item["case_id"])
    summary = build_summary(rows)
    manifest = build_manifest(
        summary=summary,
        reference_recovery_plan_jsonl=reference_recovery_plan_jsonl,
        reference_recovery_manifest_json=reference_recovery_manifest_json,
        collected_evidence_jsonl=collected_evidence_jsonl,
        missing_evidence_plan_jsonl=missing_evidence_plan_jsonl,
        hard_case_benchmark_jsonl=hard_case_benchmark_jsonl,
        golden_jsonl=golden_jsonl,
    )
    return {"rows": rows, "summary": summary, "manifest": manifest}


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Source Path Mapping Summary",
        "",
        f"- `total_rows`: {summary['total_rows']}",
        f"- `rows_needing_docker_mount_count`: {summary['rows_needing_docker_mount_count']}",
        f"- `rows_with_missing_reference_path_count`: {summary['rows_with_missing_reference_path_count']}",
        f"- `rows_with_ambiguous_mapping_count`: {summary['rows_with_ambiguous_mapping_count']}",
        f"- `can_repair_manifest_paths_count`: {summary['can_repair_manifest_paths_count']}",
        f"- `can_recover_source_audio_count`: {summary['can_recover_source_audio_count']}",
        f"- `can_recover_reference_audio_count`: {summary['can_recover_reference_audio_count']}",
        f"- `can_run_phase10e_after_mapping_count`: {summary['can_run_phase10e_after_mapping_count']}",
        "",
        "## Mapping Status",
        "",
    ]
    for key, value in summary["mapping_status_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Detected Path Prefixes", ""])
    for key, value in summary["detected_path_prefix_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Required Docker Mounts", ""])
    if summary["required_docker_mount_counts"]:
        for key, value in summary["required_docker_mount_counts"].items():
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- none")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mount_suggestions_md(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Docker Mount Suggestions",
        "",
        "This file is a plan only. It does not change `docker-compose.yml` and does not copy or read audio.",
        "",
    ]
    if not summary["required_docker_mount_counts"]:
        lines.append("No required Docker mount was detected.")
    else:
        lines.append("## Suggested Read-Only Mounts")
        lines.append("")
        for mount, count in summary["required_docker_mount_counts"].items():
            lines.append(f"- `{mount}` (`rows={count}`)")
        lines.extend(["", "## Affected Cases", ""])
        for row in rows:
            if not row.get("required_docker_mount"):
                continue
            lines.append(
                f"- `{row['case_id']}`: `{row['required_docker_mount']}`; "
                f"status=`{row['mapping_status']}`"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "source_path_mapping_plan.jsonl", result["rows"])
    (output_dir / "source_path_mapping_manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary_md(output_dir / "source_path_mapping_summary.md", result["summary"])
    write_mount_suggestions_md(output_dir / "docker_mount_suggestions.md", result["rows"], result["summary"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan Docker-visible source path mapping and manifest path repair.")
    parser.add_argument("--reference-recovery-plan-jsonl", type=Path, required=True)
    parser.add_argument("--reference-recovery-manifest-json", type=Path, required=True)
    parser.add_argument("--collected-evidence-jsonl", type=Path, required=True)
    parser.add_argument("--missing-evidence-plan-jsonl", type=Path, required=True)
    parser.add_argument("--hard-case-benchmark-jsonl", type=Path, required=True)
    parser.add_argument("--golden-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_source_path_mapping_plan(
        reference_recovery_plan_jsonl=args.reference_recovery_plan_jsonl,
        reference_recovery_manifest_json=args.reference_recovery_manifest_json,
        collected_evidence_jsonl=args.collected_evidence_jsonl,
        missing_evidence_plan_jsonl=args.missing_evidence_plan_jsonl,
        hard_case_benchmark_jsonl=args.hard_case_benchmark_jsonl,
        golden_jsonl=args.golden_jsonl,
    )
    write_outputs(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
