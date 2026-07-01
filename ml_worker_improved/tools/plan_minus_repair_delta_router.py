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

from docker_runtime import cli_runtime_contract, require_docker_runtime  # noqa: E402


SCHEMA = "minus_repair_delta_router_plan.v1"
MANIFEST_NAME = "plan.json"
ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
START_TIME = time.monotonic()

UNSAFE_ISSUES = {"vocal_projection_proxy", "vocal_stem_corr_proxy", "full_mix_like_risk_proxy"}
DETAIL_ISSUES = {"air_loss_proxy", "presence_loss_proxy", "low_rms_muffling_proxy"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def progress(progress_path: Path, phase: str, **fields: Any) -> None:
    payload = {
        "phase": phase,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "route": "minus_repair_delta_router",
        "acceleration_mode": "external_only_cpu_metadata_planning",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "router reads existing delta-gate manifests and JSONL rows only",
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


def stable_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def issues_from(row: dict[str, Any], key: str) -> set[str]:
    value = row.get(key)
    if not isinstance(value, list):
        return set()
    return {str(issue) for issue in value if str(issue)}


def load_delta_gate(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    try:
        manifest = read_json(path)
    except FileNotFoundError:
        return None, [], ["delta_gate_manifest_missing"]
    except (json.JSONDecodeError, OSError):
        return None, [], ["delta_gate_manifest_corrupt"]

    if manifest.get("schema") != "minus_repair_delta_gate.v1":
        failures = ["delta_gate_schema_unexpected"]
    else:
        failures = []
    rows_path = canonical_path(manifest.get("rows_jsonl"), path.parent)
    if not manifest.get("rows_jsonl") or not rows_path.exists():
        failures.append("delta_rows_jsonl_missing")
        return manifest, [], failures
    try:
        rows = list(iter_jsonl(rows_path))
    except (json.JSONDecodeError, OSError):
        failures.append("delta_rows_jsonl_corrupt")
        return manifest, [], failures
    if not rows:
        failures.append("delta_rows_jsonl_empty")
    return manifest, rows, failures


def issue_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    transition_counts: dict[str, int] = {}
    issue_counts: dict[str, int] = {}
    persisting_unsafe = 0
    persisting_detail = 0
    mixed = 0
    for row in rows:
        transition = str(row.get("transition") or "unknown")
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
        unresolved = transition in {"persisting_blocker", "new_blocker"}
        issues = issues_from(row, "after_issues") or issues_from(row, "persisting_issues")
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        if unresolved:
            has_unsafe = bool(issues & UNSAFE_ISSUES)
            has_detail = bool(issues & DETAIL_ISSUES)
            if has_unsafe:
                persisting_unsafe += 1
            if has_detail:
                persisting_detail += 1
            if has_unsafe and has_detail:
                mixed += 1
    return {
        "transition_counts": dict(sorted(transition_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "persistent_unsafe_blocker_segments": persisting_unsafe,
        "persistent_detail_blocker_segments": persisting_detail,
        "persistent_mixed_blocker_segments": mixed,
    }


def select_route(
    *,
    gate_manifest: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    source_route: str,
) -> tuple[str, str, str, list[str], list[str], list[str]]:
    summary = issue_summary(rows)
    gate_status = str(gate_manifest.get("gate_status") or "")
    gate_failures = [str(item) for item in gate_manifest.get("failure_taxonomy") or [] if str(item)]
    unsafe_segments = int(summary["persistent_unsafe_blocker_segments"])
    detail_segments = int(summary["persistent_detail_blocker_segments"])
    persisting = int(summary["transition_counts"].get("persisting_blocker", 0))

    if gate_status == "passed_for_candidate_routing":
        return (
            "candidate_ready_for_downstream_gates",
            "run_downstream_acceptance_gates",
            "",
            [
                "run_no_reference_result_judge_minus",
                "run_no_reference_candidate_selector_minus",
                "build_no_reference_minus_listen_preview_manifest",
            ],
            [],
            [],
        )
    if gate_status == "blocked_unsafe_risk_increase" or unsafe_segments > 0:
        return (
            "ready_for_next_candidate_route",
            "switch_to_leakage_repair",
            "judge_guided_minus_vocal_leakage_repair",
            [
                "run_judge_guided_minus_vocal_leakage_repair",
                "rerun_instrumental_self_judge_gate",
                "rerun_minus_repair_delta_gate",
            ],
            stable_unique(gate_failures + ["persistent_unsafe_vocal_or_full_mix_blockers"]),
            stable_unique([source_route] if source_route else []),
        )
    if detail_segments > 0:
        blocked = f"{source_route}:repeat_same_parameters" if source_route else "repeat_same_detail_repair_parameters"
        return (
            "ready_for_next_candidate_route",
            "retune_detail_repair",
            "judge_guided_minus_air_detail_repair",
            [
                "retune_judge_guided_minus_air_detail_repair_parameters",
                "rerun_instrumental_self_judge_gate",
                "rerun_minus_repair_delta_gate",
            ],
            stable_unique(gate_failures + ["persistent_detail_loss_blockers"]),
            [blocked],
        )
    if persisting > 0:
        return (
            "ready_for_next_candidate_route",
            "switch_to_segment_router",
            "run_no_reference_minus_segment_switch_candidate",
            [
                "run_no_reference_minus_segment_switch_candidate",
                "rerun_instrumental_self_judge_gate",
                "rerun_minus_repair_delta_gate",
            ],
            stable_unique(gate_failures + ["persistent_unclassified_blockers"]),
            stable_unique([source_route] if source_route else []),
        )
    return (
        "ready_for_next_candidate_route",
        "route_to_alternative_model_candidate",
        "no_reference_minus_candidate_model_router",
        [
            "mine_alternative_minus_model_candidate",
            "rerun_instrumental_self_judge_gate",
            "rerun_minus_repair_delta_gate",
        ],
        stable_unique(gate_failures + ["delta_gate_blocked_without_persistent_segment_evidence"]),
        stable_unique([source_route] if source_route else []),
    )


def summary_text(manifest: dict[str, Any]) -> str:
    lines = [
        "# Minus Repair Delta Router Plan",
        "",
        f"- router_status: `{manifest['router_status']}`",
        f"- selected_next_action: `{manifest['selected_next_action']}`",
        f"- selected_route: `{manifest['selected_route']}`",
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


def build_plan_row(
    *,
    candidate_id: str,
    router_status: str,
    selected_next_action: str,
    selected_route: str,
    required_actions: Sequence[str],
    failure_taxonomy: Sequence[str],
    blocked_repeat_routes: Sequence[str],
    summary: dict[str, Any],
    delta_gate_manifest: Path,
    source_route: str,
) -> dict[str, Any]:
    return {
        "schema": "minus_repair_delta_router_item.v1",
        "queue_item_id": "dr001",
        "candidate_id": candidate_id,
        "stem_type": "minus",
        "route_intent": "minus_repair_delta_router",
        "action_status": router_status,
        "next_action": selected_next_action,
        "selected_route": selected_route,
        "source_route": source_route,
        "blocked_repeat_routes": list(blocked_repeat_routes),
        "required_actions": list(required_actions),
        "failure_taxonomy": list(failure_taxonomy),
        "issue_summary": summary,
        "delta_gate_manifest": str(delta_gate_manifest),
        "delivery_allowed": False,
        "can_publish_as_result": False,
        "release_ready": False,
        "training_started": False,
    }


def plan_minus_repair_delta_router(
    *,
    delta_gate_manifest: Path,
    output_dir: Path,
    source_route: str = "",
    candidate_id: str | None = None,
    command_args: list[str] | None = None,
    runtime_environment: str = "docker",
    native_runtime_allowed: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", delta_gate_manifest=str(delta_gate_manifest))

    gate_manifest, rows, evidence_failures = load_delta_gate(delta_gate_manifest)
    if gate_manifest is None:
        gate_manifest = {}
    resolved_candidate_id = str(candidate_id or gate_manifest.get("candidate_id") or "minus_delta_candidate")
    summary = issue_summary(rows)
    if evidence_failures:
        router_status = "halt_missing_evidence"
        selected_next_action = "halt_missing_evidence"
        selected_route = ""
        required_actions = ["provide_valid_minus_repair_delta_gate"]
        failure_taxonomy = stable_unique([str(item) for item in gate_manifest.get("failure_taxonomy") or []] + evidence_failures)
        blocked_repeat_routes = stable_unique([source_route] if source_route else [])
    else:
        (
            router_status,
            selected_next_action,
            selected_route,
            required_actions,
            failure_taxonomy,
            blocked_repeat_routes,
        ) = select_route(gate_manifest=gate_manifest, rows=rows, source_route=source_route)

    row = build_plan_row(
        candidate_id=resolved_candidate_id,
        router_status=router_status,
        selected_next_action=selected_next_action,
        selected_route=selected_route,
        required_actions=required_actions,
        failure_taxonomy=failure_taxonomy,
        blocked_repeat_routes=blocked_repeat_routes,
        summary=summary,
        delta_gate_manifest=delta_gate_manifest,
        source_route=source_route,
    )
    manifest = {
        "schema": SCHEMA,
        "router_status": router_status,
        "candidate_id": resolved_candidate_id,
        "stem_type": "minus",
        "mode": "minus_repair_delta_router",
        "delta_gate_manifest": str(delta_gate_manifest),
        "source_route": source_route,
        "selected_next_action": selected_next_action,
        "selected_route": selected_route,
        "blocked_repeat_routes": list(blocked_repeat_routes),
        "required_actions": list(required_actions),
        "failure_taxonomy": list(failure_taxonomy),
        "issue_summary": summary,
        "rows": [row],
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "delivery_allowed": False,
        "delivery_status": "router_plan_only",
        "presentation_status": "router_plan_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_downstream_acceptance": True,
        "expected_acceleration_mode": "external_only_cpu_metadata_planning",
        "runtime_environment": str(runtime_environment),
        "native_runtime_allowed": bool(native_runtime_allowed),
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "router reads existing delta-gate manifests and JSONL rows only",
        "artifact_name_policy": {
            "short_output_names": True,
            "raw_ids_in_paths": False,
            "path_segment_limit": 32,
            "output_files": [MANIFEST_NAME, ROWS_NAME, SUMMARY_NAME, PROGRESS_NAME],
        },
        "command_args": command_args or [],
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    write_jsonl(output_dir / ROWS_NAME, [row])
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest), encoding="utf-8")
    progress(
        progress_path,
        "finish",
        router_status=router_status,
        selected_next_action=selected_next_action,
        selected_route=selected_route,
        failure_taxonomy=failure_taxonomy,
        manifest_path=str(output_dir / MANIFEST_NAME),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-gate-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-route", default="")
    parser.add_argument("--candidate-id")
    parser.add_argument("--short-output-names", action="store_true")
    parser.add_argument(
        "--allow-native-runtime",
        action="store_true",
        help="Explicitly allow native Mac CPU-DSP diagnostic execution outside Docker.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = cli_runtime_contract(allow_native_runtime=bool(args.allow_native_runtime))
    require_docker_runtime(
        tool_name="plan_minus_repair_delta_router.py",
        entrypoint_path="/app/tools/plan_minus_repair_delta_router.py",
        allow_native_macos_diagnostic=bool(args.allow_native_runtime),
    )
    manifest = plan_minus_repair_delta_router(
        delta_gate_manifest=args.delta_gate_manifest,
        output_dir=args.output_dir,
        source_route=args.source_route,
        candidate_id=args.candidate_id,
        command_args=list(argv if argv is not None else sys.argv[1:]),
        runtime_environment=str(runtime["runtime_environment"]),
        native_runtime_allowed=bool(runtime["native_runtime_allowed"]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "router_status": manifest["router_status"],
                "selected_next_action": manifest["selected_next_action"],
                "selected_route": manifest["selected_route"],
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
