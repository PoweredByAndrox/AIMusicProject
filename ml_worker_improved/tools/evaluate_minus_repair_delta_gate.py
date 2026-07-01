from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402


SCHEMA = "minus_repair_delta_gate.v1"
MANIFEST_NAME = "gate.json"
ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
START_TIME = time.monotonic()

UNSAFE_ISSUES = {"vocal_projection_proxy", "vocal_stem_corr_proxy", "full_mix_like_risk_proxy"}
DETAIL_ISSUES = {"air_loss_proxy", "presence_loss_proxy", "low_rms_muffling_proxy"}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def progress(progress_path: Path, phase: str, **fields: Any) -> None:
    payload = {
        "phase": phase,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "route": "minus_repair_delta_gate",
        "acceleration_mode": "external_only_cpu_metadata_gate",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "delta gate compares existing self-judge manifests and JSONL rows only",
        **fields,
    }
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def canonical_path(value: Any, base_dir: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        return Path("__missing_path__")
    path = Path(text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def as_issues(row: dict[str, Any]) -> set[str]:
    issues = row.get("issues")
    if not isinstance(issues, list):
        return set()
    return {str(issue) for issue in issues if str(issue)}


def row_key(row: dict[str, Any], index: int) -> str:
    return str(row.get("segment_id") or row.get("window_id") or f"row_{index:04d}")


def issue_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for issue in as_issues(row):
            counts[issue] = counts.get(issue, 0) + 1
    return dict(sorted(counts.items()))


def issue_group_count(counts: dict[str, int], issue_group: set[str]) -> int:
    return sum(int(counts.get(issue, 0)) for issue in issue_group)


def load_self_judge_evidence(
    *,
    manifest_path: Path,
    label: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    try:
        manifest = read_json(manifest_path)
    except FileNotFoundError:
        return None, [], [f"{label}_manifest_missing"]
    except (json.JSONDecodeError, OSError):
        return None, [], [f"{label}_manifest_corrupt"]

    if manifest.get("mode") != "instrumental_self_judge_gate":
        failures.append(f"{label}_mode_not_instrumental_self_judge_gate")

    segments_path = canonical_path(manifest.get("segments_jsonl"), manifest_path.parent)
    if not manifest.get("segments_jsonl") or not segments_path.exists():
        failures.append(f"{label}_segments_jsonl_missing")
        return manifest, [], failures

    try:
        rows = list(iter_jsonl(segments_path))
    except (json.JSONDecodeError, OSError):
        failures.append(f"{label}_segments_jsonl_corrupt")
        return manifest, [], failures
    if not rows:
        failures.append(f"{label}_segments_jsonl_empty")
    return manifest, rows, failures


def manifest_blocker_count(manifest: dict[str, Any] | None, rows: Sequence[dict[str, Any]]) -> int:
    if isinstance(manifest, dict):
        overall = manifest.get("overall")
        if isinstance(overall, dict):
            try:
                return int(overall.get("blocker_count"))
            except (TypeError, ValueError):
                pass
    return sum(1 for row in rows if bool(row.get("blocker")))


def manifest_segment_count(manifest: dict[str, Any] | None, rows: Sequence[dict[str, Any]]) -> int:
    if isinstance(manifest, dict):
        overall = manifest.get("overall")
        if isinstance(overall, dict):
            try:
                return int(overall.get("segment_count"))
            except (TypeError, ValueError):
                pass
    return len(rows)


def segment_delta_rows(
    *,
    before_rows: Sequence[dict[str, Any]],
    after_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    before_by_id = {row_key(row, index): row for index, row in enumerate(before_rows)}
    after_by_id = {row_key(row, index): row for index, row in enumerate(after_rows)}
    rows: list[dict[str, Any]] = []
    for segment_id in sorted(set(before_by_id) | set(after_by_id)):
        before = before_by_id.get(segment_id, {})
        after = after_by_id.get(segment_id, {})
        before_issues = as_issues(before)
        after_issues = as_issues(after)
        before_blocker = bool(before.get("blocker"))
        after_blocker = bool(after.get("blocker"))
        if before_blocker and after_blocker:
            transition = "persisting_blocker"
        elif before_blocker and not after_blocker:
            transition = "resolved_blocker"
        elif not before_blocker and after_blocker:
            transition = "new_blocker"
        else:
            transition = "clean"
        rows.append(
            {
                "schema": "minus_repair_delta_gate_segment.v1",
                "segment_id": segment_id,
                "transition": transition,
                "before_blocker": before_blocker,
                "after_blocker": after_blocker,
                "before_issues": sorted(before_issues),
                "after_issues": sorted(after_issues),
                "resolved_issues": sorted(before_issues - after_issues),
                "new_issues": sorted(after_issues - before_issues),
                "persisting_issues": sorted(before_issues & after_issues),
                "unsafe_issue_delta": len(after_issues & UNSAFE_ISSUES) - len(before_issues & UNSAFE_ISSUES),
                "detail_issue_delta": len(after_issues & DETAIL_ISSUES) - len(before_issues & DETAIL_ISSUES),
            }
        )
    return rows


def summarize_transition_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {
        "persisting_blocker_count": 0,
        "resolved_blocker_count": 0,
        "new_blocker_count": 0,
        "clean_count": 0,
    }
    for row in rows:
        key = f"{row['transition']}_count"
        counts[key] = counts.get(key, 0) + 1
    return counts


def load_repair_manifest(path: Path | None) -> tuple[dict[str, Any] | None, list[str]]:
    if path is None:
        return None, []
    try:
        return read_json(path), []
    except FileNotFoundError:
        return None, ["repair_manifest_missing"]
    except (json.JSONDecodeError, OSError):
        return None, ["repair_manifest_corrupt"]


def status_from_failures(failures: list[str]) -> str:
    if any("corrupt" in failure for failure in failures):
        return "blocked_corrupt_evidence"
    return "blocked_missing_evidence"


def summary_text(manifest: dict[str, Any]) -> str:
    lines = [
        "# Minus Repair Delta Gate",
        "",
        f"- gate_status: `{manifest['gate_status']}`",
        f"- before_blocker_count: `{manifest['delta']['before_blocker_count']}`",
        f"- after_blocker_count: `{manifest['delta']['after_blocker_count']}`",
        f"- blocker_count_delta: `{manifest['delta']['blocker_count_delta']}`",
        f"- unsafe_issue_count_delta: `{manifest['delta']['unsafe_issue_count_delta']}`",
        f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
        "",
        "## Failure Taxonomy",
        "",
    ]
    if manifest["failure_taxonomy"]:
        lines.extend(f"- `{failure}`" for failure in manifest["failure_taxonomy"])
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def evaluate_minus_repair_delta_gate(
    *,
    before_self_judge_manifest: Path,
    after_self_judge_manifest: Path,
    output_dir: Path,
    candidate_id: str,
    repair_manifest: Path | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", candidate_id=candidate_id)

    before_manifest, before_rows, before_failures = load_self_judge_evidence(
        manifest_path=before_self_judge_manifest,
        label="before",
    )
    after_manifest, after_rows, after_failures = load_self_judge_evidence(
        manifest_path=after_self_judge_manifest,
        label="after",
    )
    repair_payload, repair_failures = load_repair_manifest(repair_manifest)
    evidence_failures = before_failures + after_failures + repair_failures

    delta_rows = segment_delta_rows(before_rows=before_rows, after_rows=after_rows)
    before_counts = issue_counts(before_rows)
    after_counts = issue_counts(after_rows)
    all_issues = sorted(set(before_counts) | set(after_counts))
    issue_count_delta = {issue: int(after_counts.get(issue, 0)) - int(before_counts.get(issue, 0)) for issue in all_issues}
    before_blockers = manifest_blocker_count(before_manifest, before_rows)
    after_blockers = manifest_blocker_count(after_manifest, after_rows)
    before_segments = manifest_segment_count(before_manifest, before_rows)
    after_segments = manifest_segment_count(after_manifest, after_rows)
    before_unsafe = issue_group_count(before_counts, UNSAFE_ISSUES)
    after_unsafe = issue_group_count(after_counts, UNSAFE_ISSUES)
    before_detail = issue_group_count(before_counts, DETAIL_ISSUES)
    after_detail = issue_group_count(after_counts, DETAIL_ISSUES)
    delta = {
        "before_blocker_count": before_blockers,
        "after_blocker_count": after_blockers,
        "blocker_count_delta": after_blockers - before_blockers,
        "before_segment_count": before_segments,
        "after_segment_count": after_segments,
        "segment_count_delta": after_segments - before_segments,
        "before_unsafe_issue_count": before_unsafe,
        "after_unsafe_issue_count": after_unsafe,
        "unsafe_issue_count_delta": after_unsafe - before_unsafe,
        "before_detail_issue_count": before_detail,
        "after_detail_issue_count": after_detail,
        "detail_issue_count_delta": after_detail - before_detail,
        "issue_count_delta": issue_count_delta,
        **summarize_transition_counts(delta_rows),
    }

    if evidence_failures:
        gate_status = status_from_failures(evidence_failures)
        failure_taxonomy = evidence_failures
    elif delta["blocker_count_delta"] >= 0:
        gate_status = "blocked_no_blocker_reduction"
        failure_taxonomy = ["blocker_count_not_reduced"]
    elif delta["unsafe_issue_count_delta"] > 0:
        gate_status = "blocked_unsafe_risk_increase"
        failure_taxonomy = ["unsafe_issue_count_increased"]
    else:
        gate_status = "passed_for_candidate_routing"
        failure_taxonomy = []

    progress(
        progress_path,
        "evidence_loaded",
        before_blocker_count=before_blockers,
        after_blocker_count=after_blockers,
        before_segment_count=before_segments,
        after_segment_count=after_segments,
        evidence_failures=evidence_failures,
    )

    manifest = {
        "schema": SCHEMA,
        "gate_status": gate_status,
        "candidate_id": str(candidate_id),
        "stem_type": "minus",
        "mode": "minus_repair_delta_gate",
        "before_self_judge_manifest": str(before_self_judge_manifest),
        "after_self_judge_manifest": str(after_self_judge_manifest),
        "repair_manifest": str(repair_manifest) if repair_manifest is not None else None,
        "repair_manifest_delivery_allowed": repair_payload.get("delivery_allowed") if isinstance(repair_payload, dict) else None,
        "delta": delta,
        "failure_taxonomy": failure_taxonomy,
        "failed_checks": list(failure_taxonomy),
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "delivery_allowed": False,
        "delivery_status": "candidate_routing_only" if gate_status == "passed_for_candidate_routing" else "blocked",
        "presentation_status": "delta_gate_not_for_listen_pack",
        "release_status": "not_release_ready",
        "listenable": False,
        "deliverable": False,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_downstream_acceptance": True,
        "requires_post_render_minus_self_judge": True,
        "requires_no_reference_result_judge": True,
        "requires_selector_acceptance": True,
        "expected_acceleration_mode": "external_only_cpu_metadata_gate",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "delta gate compares existing self-judge manifests and JSONL rows only",
        "artifact_name_policy": {
            "short_output_names": True,
            "raw_ids_in_paths": False,
            "path_segment_limit": 32,
            "output_files": [MANIFEST_NAME, ROWS_NAME, SUMMARY_NAME, PROGRESS_NAME],
        },
        "route_provenance": {
            "before_candidate_label": (before_manifest or {}).get("candidate", {}).get("label")
            if isinstance((before_manifest or {}).get("candidate"), dict)
            else None,
            "after_candidate_label": (after_manifest or {}).get("candidate", {}).get("label")
            if isinstance((after_manifest or {}).get("candidate"), dict)
            else None,
            "evaluator": "evaluate_minus_repair_delta_gate.py",
            "command_args": command_args or [],
        },
        "next_action": (
            "run_downstream_acceptance_gates"
            if gate_status == "passed_for_candidate_routing"
            else "retune_repair_or_route_alternative"
        ),
    }

    write_json(output_dir / MANIFEST_NAME, manifest)
    write_jsonl(output_dir / ROWS_NAME, delta_rows)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest), encoding="utf-8")
    progress(
        progress_path,
        "finish",
        gate_status=gate_status,
        blocker_count_delta=delta["blocker_count_delta"],
        unsafe_issue_count_delta=delta["unsafe_issue_count_delta"],
        failure_taxonomy=failure_taxonomy,
        manifest_path=str(output_dir / MANIFEST_NAME),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-self-judge-manifest", type=Path, required=True)
    parser.add_argument("--after-self-judge-manifest", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--short-output-names", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="evaluate_minus_repair_delta_gate.py",
        entrypoint_path="/app/tools/evaluate_minus_repair_delta_gate.py",
    )
    args = parse_args(argv)
    manifest = evaluate_minus_repair_delta_gate(
        before_self_judge_manifest=args.before_self_judge_manifest,
        after_self_judge_manifest=args.after_self_judge_manifest,
        repair_manifest=args.repair_manifest,
        output_dir=args.output_dir,
        candidate_id=args.candidate_id,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "gate_status": manifest["gate_status"],
                "blocker_count_delta": manifest["delta"]["blocker_count_delta"],
                "unsafe_issue_count_delta": manifest["delta"]["unsafe_issue_count_delta"],
                "delivery_allowed": manifest["delivery_allowed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
