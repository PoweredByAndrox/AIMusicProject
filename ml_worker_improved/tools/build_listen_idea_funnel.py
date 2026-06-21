#!/usr/bin/env python3
"""Build a deterministic idea funnel from short-listen evidence artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402

MANIFEST_NAME = "funnel.json"
IDEAS_NAME = "ideas.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
START_TIME = time.monotonic()
BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "can_publish_as_result",
    "training_started",
    "separator_training_started",
    "models_changed",
    "model_registry_edited",
    "thresholds_relaxed",
    "cleanup_routes_added",
)


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_manifest_build",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "listen_idea_funnel_reads_existing_artifacts_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "listen_idea_funnel_no_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def collect_true_flag_failures(prefix: str, payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for key in BLOCKED_TRUE_FLAGS:
        if payload.get(key) is True:
            failed.append(f"{prefix}_{key}_true")
    safety = payload.get("safety")
    if isinstance(safety, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if safety.get(key) is True:
                failed.append(f"{prefix}_safety_{key}_true")
    contract = payload.get("contract")
    if isinstance(contract, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if contract.get(key) is True:
                failed.append(f"{prefix}_contract_{key}_true")
    return failed


def safety_flags() -> dict[str, bool]:
    return {
        "reference_required": False,
        "human_csv_required": False,
        "training_started": False,
        "separator_training_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
        "release_ready": False,
    }


def checks_by_id(goal_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id") or ""): row for row in goal_audit.get("checks") or [] if isinstance(row, dict)}


def stem_coverage(pack: dict[str, Any]) -> dict[str, int]:
    coverage: dict[str, int] = {}
    for row in pack.get("items") or pack.get("listen_audio_files") or []:
        if not isinstance(row, dict):
            continue
        stem = str(row.get("stem_type") or "unknown")
        coverage[stem] = coverage.get(stem, 0) + 1
    return coverage


def resolve_rows_jsonl(raw: Any, *, base: Path) -> Path:
    path = Path(str(raw or ""))
    if not str(path):
        return path
    if path.is_absolute():
        return path
    return base / path


def minus_audit_warning_evidence(minus_audit_json: Path | None, minus_audit: dict[str, Any]) -> dict[str, Any]:
    if not minus_audit_json or not minus_audit:
        return {}
    rows_path = resolve_rows_jsonl(minus_audit.get("rows_jsonl"), base=minus_audit_json.parent)
    rows = read_jsonl(rows_path)
    warning_rows = [row for row in rows if row.get("quality_warning_reasons")]
    blocker_rows = [row for row in rows if row.get("audit_blockers")]
    warning_reasons = stable_unique(
        reason for row in warning_rows for reason in row.get("quality_warning_reasons") or []
    )
    warning_audio_paths = stable_unique(row.get("short_audio_path") for row in warning_rows)
    blocker_reasons = stable_unique(reason for row in blocker_rows for reason in row.get("audit_blockers") or [])
    blocked_audio_paths = stable_unique(row.get("short_audio_path") for row in blocker_rows)
    return {
        "minus_audit_json": str(minus_audit_json),
        "audit_status": minus_audit.get("audit_status"),
        "accepted_minus_count": minus_audit.get("accepted_minus_count"),
        "blocked_minus_count": minus_audit.get("blocked_minus_count"),
        "evidence_missing_count": minus_audit.get("evidence_missing_count"),
        "warning_count": minus_audit.get("warning_count"),
        "warning_reasons": warning_reasons,
        "warning_audio_paths": warning_audio_paths,
        "blocker_reasons": blocker_reasons,
        "blocked_audio_paths": blocked_audio_paths,
    }


def idea(
    *,
    index: int,
    slug: str,
    title: str,
    rationale: str,
    evidence: dict[str, Any],
    docker_command: str,
    manual_required: bool = False,
    expected_output: str = "",
) -> dict[str, Any]:
    return {
        "schema": "short_listen_idea_funnel_item.v1",
        "rank": index,
        "idea_id": f"i{index:03d}_{slug}",
        "title": title,
        "rationale": rationale,
        "evidence": evidence,
        "docker_command": docker_command,
        "manual_required": manual_required,
        "expected_output": expected_output,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
        "separator_training_started": False,
        "thresholds_relaxed": False,
        "models_changed": False,
        "model_registry_edited": False,
    }


def summary_text(manifest: dict[str, Any], ideas: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# listen idea funnel",
        "",
        f"- funnel_status: `{manifest['funnel_status']}`",
        f"- idea_count: `{manifest['idea_count']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        "",
        "## Ideas",
        "",
    ]
    for row in ideas:
        lines.append(f"- `{row['idea_id']}` {row['title']}")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    return "\n".join(lines) + "\n"


def build_ideas(
    *,
    goal_audit_json: Path,
    pack_manifest_json: Path,
    goal_audit: dict[str, Any],
    pack_manifest: dict[str, Any],
    minus_cmp_json: Path | None,
    minus_cmp: dict[str, Any],
    targeted_cmp_json: Path | None,
    targeted_cmp: dict[str, Any],
    minus_audit_json: Path | None,
    minus_audit: dict[str, Any],
    feedback_manifest_json: Path | None,
    feedback_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    checks = checks_by_id(goal_audit)
    coverage = stem_coverage(pack_manifest)
    rows: list[dict[str, Any]] = []

    if int(coverage.get("minus") or 0) > 0:
        rows.append(
            idea(
                index=len(rows) + 1,
                slug="run_short_minus_quality_audit",
                title="Audit selected minus items for residual vocal and muffling focus",
                rationale=(
                    "The listen goal includes reported minus muffling and residual-vocal risk; "
                    "selected minus rows need a reproducible post-pack no-reference audit before "
                    "their listening verdict is treated as final evidence."
                ),
                evidence={
                    "pack_manifest_json": str(pack_manifest_json),
                    "minus_item_count": coverage.get("minus"),
                    "stem_coverage": coverage,
                },
                docker_command=(
                    "docker compose run --rm --no-deps ml-worker python "
                    "tools/build_short_minus_quality_audit.py "
                    f"--pack-manifest-json {pack_manifest_json} "
                    "--output-dir <minus_quality_audit_dir> "
                    "--window-seconds 12 "
                    "--max-windows-per-candidate 5"
                ),
                expected_output="minus_quality_audit_dir/audit.json",
            )
        )
        rows.append(
            idea(
                index=len(rows) + 1,
                slug="build_minus_audit_gated_subset",
                title="Rebuild the listen subset without blocked minus rows",
                rationale=(
                    "Minus audit rows must be consumed by the next short listen artifact so blocked "
                    "or evidence-missing minus items are not presented as accepted listen rows."
                ),
                evidence={
                    "pack_manifest_json": str(pack_manifest_json),
                    "minus_audit_input": "<minus_quality_audit_dir>/audit.json",
                    "minus_item_count": coverage.get("minus"),
                },
                docker_command=(
                    "docker compose run --rm --no-deps ml-worker python "
                    "tools/build_short_listen_subset_pack.py "
                    f"--source-manifest-json {pack_manifest_json} "
                    "--minus-audit-json <minus_quality_audit_dir>/audit.json "
                    "--output-dir <audit_gated_subset_dir> "
                    "--alias-prefix g"
                ),
                expected_output="audit_gated_subset_dir/pack.json",
            )
        )
        rows.append(
            idea(
                index=len(rows) + 1,
                slug="build_minus_warning_repair_queue",
                title="Route accepted-with-warning minus rows into repair queue",
                rationale=(
                    "Minus rows that pass gates with residual-vocal or detail-loss warnings are still "
                    "not clean production evidence; they need a machine-warning repair/mining queue "
                    "without pretending that a human rejected them."
                ),
                evidence={
                    "pack_manifest_json": str(pack_manifest_json),
                    "minus_audit_input": "<minus_quality_audit_dir>/audit.json",
                    "queue_source": "minus_audit_warning",
                },
                docker_command=(
                    "docker compose run --rm --no-deps ml-worker python "
                    "tools/build_short_listen_repair_queue.py "
                    "--minus-audit-json <minus_quality_audit_dir>/audit.json "
                    "--output-dir <minus_warning_queue_dir>"
                ),
                expected_output="minus_warning_queue_dir/queue.json",
            )
        )

    warning_evidence = minus_audit_warning_evidence(minus_audit_json, minus_audit)
    if int(warning_evidence.get("blocked_minus_count") or 0) > 0 or warning_evidence.get("blocker_reasons"):
        rows.append(
            idea(
                index=len(rows) + 1,
                slug="mine_current_minus_blocker_rows",
                title="Mine current blocked minus rows before listen completion",
                rationale=(
                    "The latest minus audit still has hard blocked rows; those exact short paths "
                    "must drive the next repair/mining cycle before the listen pack can be treated "
                    "as clean machine evidence."
                ),
                evidence=warning_evidence,
                docker_command=(
                    "docker compose run --rm --no-deps ml-worker python "
                    "tools/build_short_listen_repair_queue.py "
                    f"--minus-audit-json {minus_audit_json} "
                    "--output-dir <minus_blocker_queue_dir>"
                ),
                expected_output="minus_blocker_queue_dir/queue.json",
            )
        )
    if int(warning_evidence.get("warning_count") or 0) > 0:
        rows.append(
            idea(
                index=len(rows) + 1,
                slug="mine_current_minus_warning_rows",
                title="Mine current accepted-with-warning minus rows",
                rationale=(
                    "The latest minus audit passed hard blockers but still reported warning rows; "
                    "those exact short paths must feed the next repair/mining cycle instead of being "
                    "treated as clean completion evidence."
                ),
                evidence=warning_evidence,
                docker_command=(
                    "docker compose run --rm --no-deps ml-worker python "
                    "tools/build_short_listen_repair_queue.py "
                    f"--minus-audit-json {minus_audit_json} "
                    "--output-dir <minus_warning_queue_dir>"
                ),
                expected_output="minus_warning_queue_dir/queue.json",
            )
        )
        rows.append(
            idea(
                index=len(rows) + 1,
                slug="run_minus_replacement_sweep",
                title="Evaluate accepted minus repair candidates against current warnings",
                rationale=(
                    "Accepted-with-warning minus rows need a reproducible replacement sweep "
                    "before any candidate can be forwarded for listen review; the sweep must "
                    "prove warning or blocker improvement instead of selecting the least-bad "
                    "candidate by hand."
                ),
                evidence=warning_evidence,
                docker_command=(
                    "docker compose run --rm --no-deps ml-worker python "
                    "tools/build_short_minus_replacement_sweep.py "
                    f"--base-pack-json {pack_manifest_json} "
                    "--repair-candidate-pack-json <minus_repair_candidate_pack_dir>/pack.json "
                    "--minus-result-judge-jsonl <minus_result_judge.jsonl> "
                    "--minus-windowed-judge-jsonl <minus_windowed_judge.jsonl> "
                    f"--baseline-audit-json {minus_audit_json} "
                    "--output-dir <minus_replacement_sweep_dir>"
                ),
                expected_output="minus_replacement_sweep_dir/sweep.json",
            )
        )

    human_check = checks.get("human_listen_verdict", {})
    human_blockers = human_check.get("blockers") if isinstance(human_check, dict) else []
    if str(goal_audit.get("goal_status") or "") == "machine_pass_human_listen_pending" or human_blockers:
        rows.append(
            idea(
                index=len(rows) + 1,
                slug="collect_human_listen_verdict",
                title="Collect short-path listen verdict before claiming completion",
                rationale="Machine gates passed but the goal audit still blocks on missing human listen verdict.",
                evidence={
                    "goal_audit_json": str(goal_audit_json),
                    "goal_status": goal_audit.get("goal_status"),
                    "goal_status_reason": goal_audit.get("goal_status_reason"),
                    "human_blockers": human_blockers or [],
                },
                docker_command=(
                    "docker compose run --rm --no-deps ml-worker python "
                    "tools/run_short_listen_feedback_cycle.py "
                    f"--pack-manifest-json {pack_manifest_json} "
                    "--aggregate-validation-json <validation.json> "
                    "--feedback-jsonl <filled_items.jsonl> "
                    "--output-dir <feedback_cycle_dir>"
                ),
                manual_required=True,
                expected_output="feedback_cycle_*/cycle.json",
            )
        )

    if targeted_cmp and int(targeted_cmp.get("regressed_count") or 0) == 0:
        rows.append(
            idea(
                index=len(rows) + 1,
                slug="keep_segment_switch_for_minus",
                title="Keep segment-aware full-track switch as the next minus family",
                rationale="Full replacement had mixed window results, while targeted full-track switch removed regressed windows.",
                evidence={
                    "minus_cmp_json": str(minus_cmp_json or ""),
                    "minus_cmp_improved_count": minus_cmp.get("improved_count"),
                    "minus_cmp_regressed_count": minus_cmp.get("regressed_count"),
                    "targeted_cmp_json": str(targeted_cmp_json or ""),
                    "targeted_cmp_regressed_count": targeted_cmp.get("regressed_count"),
                },
                docker_command=(
                    "docker compose run --rm --no-deps ml-worker python "
                    "tools/build_fts_listen_pack.py "
                    "--switch-manifest-json <fts_manifest.json> "
                    "--result-judge-jsonl <result_judge.jsonl> "
                    "--windowed-judge-jsonl <windowed_judge.jsonl> "
                    "--targeted-cmp-json <targeted_cmp.json> "
                    "--output-dir <full_switch_pack_dir>"
                ),
                expected_output="full_track_switch_listen_pack.v1",
            )
        )

    rows.append(
        idea(
            index=len(rows) + 1,
            slug="expand_multi_track_short_pack_cycle",
            title="Run the short-name listen cycle across more real tracks",
            rationale="The current pack proves packaging and coverage, but the objective requires a reproducible multi-track cycle.",
            evidence={
                "pack_manifest_json": str(pack_manifest_json),
                "pack_item_count": pack_manifest.get("item_count"),
                "stem_coverage": coverage,
            },
            docker_command=(
                "docker compose run --rm --no-deps ml-worker python "
                "tools/run_registered_candidate_family_wave.py "
                "--output-dir <multi_track_wave_dir> "
                "--candidate-family full_track_switch_minus "
                "--profile safe"
            ),
            expected_output="multi_track_wave_dir/manifest.json",
        )
    )

    six_stem_check = checks.get("six_stem_coverage", {})
    if isinstance(six_stem_check, dict) and six_stem_check.get("status") == "proved":
        six_evidence = six_stem_check.get("evidence") if isinstance(six_stem_check.get("evidence"), dict) else {}
        complete_groups = six_evidence.get("complete_groups") if isinstance(six_evidence.get("complete_groups"), list) else []
        source_aliases = [
            str(group.get("group_alias") or "")
            for group in complete_groups
            if isinstance(group, dict) and str(group.get("group_alias") or "")
        ]
        required_stems = six_evidence.get("required_stems") if isinstance(six_evidence.get("required_stems"), list) else []
        if not required_stems:
            required_stems = sorted(
                {
                    str(stem)
                    for group in complete_groups
                    if isinstance(group, dict)
                    for stem in (group.get("stem_types") or [])
                    if str(stem)
                }
            )
        if not required_stems:
            required_stems = ["bass", "drums", "guitar", "minus", "other", "piano", "vocal"]
        source_alias_flags = " ".join(f"--source-alias {alias}" for alias in source_aliases)
        required_stem_flags = " ".join(f"--required-stem-type {stem}" for stem in sorted({str(stem) for stem in required_stems}))
        optional_source_alias_flags = f" {source_alias_flags}" if source_alias_flags else ""
        rows.append(
            idea(
                index=len(rows) + 1,
                slug="add_semantic_stem_quality_review",
                title="Separate 6-stem coverage from semantic stem quality",
                rationale="Coverage is proved, but semantic cleanliness is not ground truth and should get its own review gate.",
                evidence={
                    "six_stem_check": six_stem_check,
                    "stem_coverage": coverage,
                },
                docker_command=(
                    "docker compose run --rm --no-deps ml-worker python "
                    "tools/build_music_intelligence_stem_listen_pack.py "
                    f"--input-pack-json {pack_manifest_json}"
                    f"{optional_source_alias_flags} "
                    f"{required_stem_flags} "
                    "--output-dir <music_stem_review_dir>"
                ),
                expected_output="music_stem_review_dir/pack.json",
            )
        )
        rows.append(
            idea(
                index=len(rows) + 1,
                slug="run_semantic_stem_batch",
                title="Run semantic proxy judge on the short 6-stem review pack",
                rationale="The 6-stem review pack is only pending until the existing semantic batch writes machine-readable verified/blocked rows.",
                evidence={
                    "semantic_pack_input": "<music_stem_review_dir>/pack.json",
                    "complete_group_aliases": source_aliases,
                    "required_stems": sorted({str(stem) for stem in required_stems}),
                },
                docker_command=(
                    "docker compose run --rm --no-deps ml-worker python "
                    "tools/run_no_reference_stem_semantic_batch.py "
                    "--input-pack-json <music_stem_review_dir>/pack.json "
                    "--short-output-names "
                    "--output-root <semantic_stem_batch_dir>"
                ),
                expected_output="semantic_stem_batch_dir/review/review.json",
            )
        )
        rows.append(
            idea(
                index=len(rows) + 1,
                slug="plan_semantic_route_gaps",
                title="Turn semantic blockers into a route-gap plan",
                rationale="Completed semantic review can still block unsupported or dirty stems; route gaps need a ranked machine-readable plan before new candidate work.",
                evidence={
                    "semantic_review_input": "<semantic_stem_batch_dir>/review/review.json",
                    "complete_group_aliases": source_aliases,
                },
                docker_command=(
                    "docker compose run --rm --no-deps ml-worker python "
                    "tools/build_semantic_stem_route_gap_plan.py "
                    "--semantic-review-json <semantic_stem_batch_dir>/review/review.json "
                    "--output-dir <semantic_route_gap_dir>"
                ),
                expected_output="semantic_route_gap_dir/gap.json",
            )
        )

    if feedback_manifest and str(feedback_manifest.get("overall_listen_verdict") or "") == "rejected":
        rows.append(
            idea(
                index=len(rows) + 1,
                slug="mine_rejected_short_paths",
                title="Mine only rejected short paths from feedback into repair queue",
                rationale="Rejected feedback should drive repair/mining without contaminating accepted items.",
                evidence={
                    "feedback_manifest_json": str(feedback_manifest_json or ""),
                    "issue_tag_counts": feedback_manifest.get("issue_tag_counts"),
                    "rejected_audio_paths": feedback_manifest.get("rejected_audio_paths"),
                },
                docker_command=(
                    "docker compose run --rm --no-deps ml-worker python "
                    "tools/build_short_listen_repair_queue.py "
                    f"--feedback-manifest-json {feedback_manifest_json} "
                    "--output-dir <repair_queue_dir>"
                ),
                expected_output="repair_queue_dir/queue.json",
            )
        )

    return rows


def build_listen_idea_funnel(
    *,
    goal_audit_json: Path,
    pack_manifest_json: Path,
    output_dir: Path,
    minus_cmp_json: Path | None = None,
    targeted_cmp_json: Path | None = None,
    minus_audit_json: Path | None = None,
    feedback_manifest_json: Path | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", goal_audit_json=str(goal_audit_json), pack_manifest_json=str(pack_manifest_json))

    goal_audit = read_json(goal_audit_json)
    pack_manifest = read_json(pack_manifest_json)
    minus_cmp = read_json(minus_cmp_json) if minus_cmp_json else {}
    targeted_cmp = read_json(targeted_cmp_json) if targeted_cmp_json else {}
    minus_audit = read_json(minus_audit_json) if minus_audit_json else {}
    feedback_manifest = read_json(feedback_manifest_json) if feedback_manifest_json else {}

    failed: list[str] = []
    failed.extend(collect_true_flag_failures("goal_audit", goal_audit))
    failed.extend(collect_true_flag_failures("pack", pack_manifest))
    failed.extend(collect_true_flag_failures("minus_cmp", minus_cmp))
    failed.extend(collect_true_flag_failures("targeted_cmp", targeted_cmp))
    failed.extend(collect_true_flag_failures("minus_audit", minus_audit))
    failed.extend(collect_true_flag_failures("feedback", feedback_manifest))
    if pack_manifest.get("schema") != "aggregate_short_listen_pack_manifest.v1":
        failed.append("pack_schema_unexpected")
    if pack_manifest.get("delivery_status") != "accepted" or pack_manifest.get("delivery_allowed") is not True:
        failed.append("pack_not_accepted")
    if minus_cmp and minus_cmp.get("comparison_status") != "completed":
        failed.append("minus_cmp_not_completed")
    if targeted_cmp and targeted_cmp.get("comparison_status") != "completed":
        failed.append("targeted_cmp_not_completed")
    if minus_audit and minus_audit.get("audit_status") not in {"completed", "completed_with_blockers"}:
        failed.append("minus_audit_not_completed")

    failed = stable_unique(failed)
    ideas = [] if failed else build_ideas(
        goal_audit_json=goal_audit_json,
        pack_manifest_json=pack_manifest_json,
        goal_audit=goal_audit,
        pack_manifest=pack_manifest,
        minus_cmp_json=minus_cmp_json,
        minus_cmp=minus_cmp,
        targeted_cmp_json=targeted_cmp_json,
        targeted_cmp=targeted_cmp,
        minus_audit_json=minus_audit_json,
        minus_audit=minus_audit,
        feedback_manifest_json=feedback_manifest_json,
        feedback_manifest=feedback_manifest,
    )
    status = "ready" if ideas and not failed else "blocked"
    manifest = {
        "schema": "short_listen_idea_funnel.v1",
        "funnel_status": status,
        "goal_audit_json": str(goal_audit_json),
        "pack_manifest_json": str(pack_manifest_json),
        "minus_cmp_json": str(minus_cmp_json or ""),
        "targeted_cmp_json": str(targeted_cmp_json or ""),
        "minus_audit_json": str(minus_audit_json or ""),
        "feedback_manifest_json": str(feedback_manifest_json or ""),
        "idea_count": len(ideas),
        "ideas_jsonl": str(output_dir / IDEAS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "failed_checks": failed,
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "idea_funnel_not_listen_result",
        "safety": safety_flags(),
        "not_claimed": ["release_ready", "publishable_result", "model_training", "human_listen_verdict"],
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / IDEAS_NAME, ideas)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest, ideas), encoding="utf-8")
    progress(progress_path, "finish", funnel_status=status, idea_count=len(ideas), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic short-listen idea funnel from artifacts.")
    parser.add_argument("--goal-audit-json", type=Path, required=True)
    parser.add_argument("--pack-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minus-cmp-json", type=Path)
    parser.add_argument("--targeted-cmp-json", type=Path)
    parser.add_argument("--minus-audit-json", type=Path)
    parser.add_argument("--feedback-manifest-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="build_listen_idea_funnel.py", entrypoint_path="/app/tools/build_listen_idea_funnel.py")
    args = parse_args(argv)
    manifest = build_listen_idea_funnel(
        goal_audit_json=args.goal_audit_json,
        pack_manifest_json=args.pack_manifest_json,
        output_dir=args.output_dir,
        minus_cmp_json=args.minus_cmp_json,
        targeted_cmp_json=args.targeted_cmp_json,
        minus_audit_json=args.minus_audit_json,
        feedback_manifest_json=args.feedback_manifest_json,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "funnel_status": manifest["funnel_status"],
                "idea_count": manifest["idea_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["funnel_status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
