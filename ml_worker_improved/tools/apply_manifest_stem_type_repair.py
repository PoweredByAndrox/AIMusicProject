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


REPAIR_STATUSES = {
    "repaired_overlay_created",
    "blocked_ambiguous_stem_type",
    "blocked_missing_evidence",
    "skipped_requires_user_file",
    "skipped_not_automatic",
}

DEFAULT_FORBIDDEN_ACTIONS = [
    "do_not_add_models",
    "do_not_add_weights",
    "do_not_start_training",
    "do_not_build_dataset",
    "do_not_build_gt_bundle",
    "do_not_copy_or_export_audio",
    "do_not_read_or_hash_audio",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_thresholds",
    "do_not_change_promotion_policy",
    "do_not_promote_artifacts",
    "do_not_mutate_source_manifests",
]

SAFETY_FLAGS = {
    "audio_copied": False,
    "audio_read": False,
    "audio_hashed": False,
    "dataset_built": False,
    "gt_bundle_built": False,
    "training_started": False,
    "policy_changed": False,
    "source_manifests_mutated": False,
    "models_changed": False,
    "weights_added": False,
    "cleanup_routes_added": False,
    "thresholds_changed": False,
}

METADATA_SUFFIXES = {".json", ".jsonl"}
UNKNOWN_STEM_VALUES = {"", "unknown", "missing", "null", "none"}


def sha256_file(path: Path) -> str:
    if path.suffix.lower() not in METADATA_SUFFIXES:
        raise ValueError(f"Refusing to hash non-metadata input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
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


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if isinstance(value, set):
        return sorted(str(item) for item in value if str(item))
    text = str(value)
    return [text] if text else []


def stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def row_id(row: dict[str, Any]) -> str:
    for key in ("case_id", "benchmark_case_id", "source_hard_case_id"):
        value = row.get(key)
        if value:
            return str(value)
    raise ValueError(f"Row is missing case id: {row}")


def rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {row_id(row): row for row in iter_jsonl(path)}


def resolve_metadata_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    if raw_path.startswith("/app/lab_artifacts/"):
        candidate = APP_ROOT / "lab_artifacts" / raw_path.removeprefix("/app/lab_artifacts/")
        if candidate.exists():
            return candidate
        host_candidate = APP_ROOT.parent / "lab_artifacts" / raw_path.removeprefix("/app/lab_artifacts/")
        if host_candidate.exists():
            return host_candidate
    if raw_path.startswith("/app/"):
        candidate = APP_ROOT / raw_path.removeprefix("/app/")
        if candidate.exists():
            return candidate
    return path


def collect_stem_candidates_from_manifest(manifest_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    payload = read_json(manifest_path)
    evidence: list[dict[str, Any]] = []
    candidates: list[str] = []

    def add_candidates(source: str, value: Any) -> None:
        values = as_list(value)
        normalized = [item for item in values if item.lower() not in UNKNOWN_STEM_VALUES]
        if normalized:
            evidence.append({"source": source, "value": normalized})
            candidates.extend(normalized)

    inputs = payload.get("inputs")
    if isinstance(inputs, dict):
        add_candidates("source_manifest.inputs.stem_names", inputs.get("stem_names"))
        add_candidates("source_manifest.inputs.target_stems", inputs.get("target_stems"))
        add_candidates("source_manifest.inputs.output_stems", inputs.get("output_stems"))
    add_candidates("source_manifest.stem_names", payload.get("stem_names"))
    add_candidates("source_manifest.target_stems", payload.get("target_stems"))
    add_candidates("source_manifest.output_stems", payload.get("output_stems"))

    return stable_unique(candidates), evidence


def infer_stem_type(row: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    source_manifest = str(row.get("source_manifest") or "")
    if not source_manifest:
        return "", "blocked_missing_evidence", [{"source": "unblock_plan.source_manifest", "value": "missing"}]
    metadata_path = resolve_metadata_path(source_manifest)
    if metadata_path.suffix.lower() not in METADATA_SUFFIXES or not metadata_path.exists():
        return "", "blocked_missing_evidence", [{"source": "source_manifest", "value": "missing_or_non_metadata"}]
    candidates, evidence = collect_stem_candidates_from_manifest(metadata_path)
    if len(candidates) == 1:
        return candidates[0], "repaired_overlay_created", evidence
    if len(candidates) > 1:
        return "", "blocked_ambiguous_stem_type", evidence
    return "", "blocked_missing_evidence", evidence or [{"source": "source_manifest", "value": "no_stem_names"}]


def merge_forbidden_actions(row: dict[str, Any], *, golden_protected: bool, holdout_protected: bool) -> list[str]:
    actions = list(DEFAULT_FORBIDDEN_ACTIONS)
    actions.extend(as_list(row.get("forbidden_actions")))
    if golden_protected:
        actions.append("golden_protected_no_training_or_tuning")
    if holdout_protected:
        actions.append("holdout_protected_no_training")
    return stable_unique(actions)


def load_golden_track_hashes(golden_jsonl: Path) -> set[str]:
    return {str(row["track_hash"]) for row in iter_jsonl(golden_jsonl) if row.get("track_hash")}


def repair_row(
    row: dict[str, Any],
    *,
    golden_track_hashes: set[str],
    evidence_rows: dict[str, dict[str, Any]],
    preflight_rows: dict[str, dict[str, Any]],
    benchmark_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    case_id = row_id(row)
    evidence_row = evidence_rows.get(case_id, {})
    preflight_row = preflight_rows.get(case_id, {})
    benchmark_row = benchmark_rows.get(case_id, {})

    source_manifest = str(
        row.get("source_manifest")
        or evidence_row.get("source_manifest")
        or benchmark_row.get("source_manifest")
        or preflight_row.get("source_manifest")
        or ""
    )
    track_hash = str(row.get("track_hash") or evidence_row.get("track_hash") or benchmark_row.get("track_hash") or "")
    split = str(row.get("split") or evidence_row.get("split") or benchmark_row.get("split") or "unknown")
    original_stem_type = str(row.get("stem_type") or evidence_row.get("stem_type") or "unknown")

    enriched = dict(row)
    enriched.update({"source_manifest": source_manifest, "track_hash": track_hash, "split": split})

    overlay_patch: dict[str, Any] = {}
    inferred_stem_type = ""
    evidence_used: list[dict[str, Any]] = []
    if row.get("requires_user_file"):
        repair_status = "skipped_requires_user_file"
    elif not row.get("can_apply_automatically"):
        repair_status = "skipped_not_automatic"
    else:
        inferred_stem_type, repair_status, evidence_used = infer_stem_type(enriched)
        if repair_status == "repaired_overlay_created":
            overlay_patch = {"op": "add_or_replace", "path": "/stem_type", "value": inferred_stem_type}

    if repair_status not in REPAIR_STATUSES:
        raise ValueError(f"Invalid repair status for {case_id}: {repair_status}")

    golden_protected = track_hash in golden_track_hashes
    holdout_protected = split == "holdout"
    return {
        "case_id": case_id,
        "track_hash": track_hash,
        "original_stem_type": original_stem_type,
        "inferred_stem_type": inferred_stem_type,
        "source_manifest": source_manifest,
        "repair_status": repair_status,
        "overlay_patch": overlay_patch,
        "evidence_used": evidence_used,
        "can_re_run_preflight_after_overlay": repair_status == "repaired_overlay_created",
        "can_start_training_now": False,
        "forbidden_actions": merge_forbidden_actions(
            enriched,
            golden_protected=golden_protected,
            holdout_protected=holdout_protected,
        ),
    }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "") for row in rows).items()))


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_repair_candidates": len(rows),
        "repair_status_counts": count_by(rows, "repair_status"),
        "overlay_repairs_created": sum(1 for row in rows if row["repair_status"] == "repaired_overlay_created"),
        "skipped_requires_user_file": sum(1 for row in rows if row["repair_status"] == "skipped_requires_user_file"),
        "skipped_not_automatic": sum(1 for row in rows if row["repair_status"] == "skipped_not_automatic"),
        "blocked_ambiguous_stem_type": sum(1 for row in rows if row["repair_status"] == "blocked_ambiguous_stem_type"),
        "blocked_missing_evidence": sum(1 for row in rows if row["repair_status"] == "blocked_missing_evidence"),
        "can_re_run_preflight_after_overlay_count": sum(1 for row in rows if row["can_re_run_preflight_after_overlay"]),
        "can_start_training_now_count": sum(1 for row in rows if row["can_start_training_now"]),
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(input_paths: dict[str, Path], rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "manifest_stem_type_repair.v1",
        "row_count": len(rows),
        "total_rows": len(rows),
        "input_files": {name: str(path) for name, path in sorted(input_paths.items())},
        "input_sha256": {name: sha256_file(path) for name, path in sorted(input_paths.items())},
        "outputs": [
            "manifest_stem_type_repair_report.jsonl",
            "manifest_stem_type_repair_manifest.json",
            "manifest_stem_type_repair_summary.md",
            "repaired_manifest_overlay.jsonl",
        ],
        "summary": summary,
        "safety": dict(SAFETY_FLAGS),
    }


def apply_manifest_stem_type_repair(
    *,
    unblock_plan_jsonl: Path,
    unblock_manifest_json: Path,
    evidence_gap_report_jsonl: Path,
    finetune_preflight_jsonl: Path,
    hard_case_benchmark_jsonl: Path,
    golden_jsonl: Path,
) -> dict[str, Any]:
    input_paths = {
        "unblock_plan_jsonl": unblock_plan_jsonl,
        "unblock_manifest_json": unblock_manifest_json,
        "evidence_gap_report_jsonl": evidence_gap_report_jsonl,
        "finetune_preflight_jsonl": finetune_preflight_jsonl,
        "hard_case_benchmark_jsonl": hard_case_benchmark_jsonl,
        "golden_jsonl": golden_jsonl,
    }
    for path in input_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in METADATA_SUFFIXES:
            raise ValueError(f"Expected metadata input, got: {path}")

    unblock_manifest = read_json(unblock_manifest_json)
    unblock_rows = list(iter_jsonl(unblock_plan_jsonl))
    manifest_row_count = unblock_manifest.get("row_count", unblock_manifest.get("total_rows"))
    if manifest_row_count is not None and int(manifest_row_count) != len(unblock_rows):
        raise ValueError(
            "unblock manifest row_count does not match JSONL rows: "
            f"{manifest_row_count} != {len(unblock_rows)}"
        )

    repair_candidates = [
        row for row in unblock_rows if row.get("unblock_action_type") == "repair_stem_type_metadata"
    ]
    evidence_rows = rows_by_id(evidence_gap_report_jsonl)
    preflight_rows = rows_by_id(finetune_preflight_jsonl)
    benchmark_rows = rows_by_id(hard_case_benchmark_jsonl)
    golden_track_hashes = load_golden_track_hashes(golden_jsonl)
    rows = [
        repair_row(
            row,
            golden_track_hashes=golden_track_hashes,
            evidence_rows=evidence_rows,
            preflight_rows=preflight_rows,
            benchmark_rows=benchmark_rows,
        )
        for row in sorted(repair_candidates, key=row_id)
    ]
    summary = build_summary(rows)
    manifest = build_manifest(input_paths, rows, summary)
    overlays = [
        {
            "case_id": row["case_id"],
            "track_hash": row["track_hash"],
            "source_manifest": row["source_manifest"],
            "overlay_patch": row["overlay_patch"],
            "evidence_used": row["evidence_used"],
        }
        for row in rows
        if row["repair_status"] == "repaired_overlay_created"
    ]
    return {"rows": rows, "overlays": overlays, "summary": summary, "manifest": manifest}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Manifest Stem Type Repair",
        "",
        "Metadata-only overlay repair. Original source manifests are not modified.",
        "",
        f"- total_repair_candidates: {summary['total_repair_candidates']}",
        f"- overlay_repairs_created: {summary['overlay_repairs_created']}",
        f"- skipped_requires_user_file: {summary['skipped_requires_user_file']}",
        f"- skipped_not_automatic: {summary['skipped_not_automatic']}",
        f"- blocked_ambiguous_stem_type: {summary['blocked_ambiguous_stem_type']}",
        f"- blocked_missing_evidence: {summary['blocked_missing_evidence']}",
        f"- can_re_run_preflight_after_overlay_count: {summary['can_re_run_preflight_after_overlay_count']}",
        f"- can_start_training_now_count: {summary['can_start_training_now_count']}",
        "",
        "## Repair Status Counts",
        "",
    ]
    for key, value in summary["repair_status_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "manifest_stem_type_repair_report.jsonl", result["rows"])
    (output_dir / "manifest_stem_type_repair_manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary_md(output_dir / "manifest_stem_type_repair_summary.md", result["summary"])
    write_jsonl(output_dir / "repaired_manifest_overlay.jsonl", result["overlays"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unblock-plan-jsonl", required=True, type=Path)
    parser.add_argument("--unblock-manifest-json", required=True, type=Path)
    parser.add_argument("--evidence-gap-report-jsonl", required=True, type=Path)
    parser.add_argument("--finetune-preflight-jsonl", required=True, type=Path)
    parser.add_argument("--hard-case-benchmark-jsonl", required=True, type=Path)
    parser.add_argument("--golden-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    result = apply_manifest_stem_type_repair(
        unblock_plan_jsonl=args.unblock_plan_jsonl,
        unblock_manifest_json=args.unblock_manifest_json,
        evidence_gap_report_jsonl=args.evidence_gap_report_jsonl,
        finetune_preflight_jsonl=args.finetune_preflight_jsonl,
        hard_case_benchmark_jsonl=args.hard_case_benchmark_jsonl,
        golden_jsonl=args.golden_jsonl,
    )
    write_outputs(result, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "total_repair_candidates": result["summary"]["total_repair_candidates"],
                "overlay_repairs_created": result["summary"]["overlay_repairs_created"],
                "can_start_training_now_count": result["summary"]["can_start_training_now_count"],
                "safety": result["summary"]["safety"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
