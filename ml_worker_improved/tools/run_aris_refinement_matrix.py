#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from artifact_name_policy import validate_artifact_relative_path, validate_artifact_segment  # noqa: E402
from docker_runtime import require_docker_runtime  # noqa: E402

FORBIDDEN_PRIMARY_DIRECTIONS = {
    "qwen_controller",
    "diffusion_generative_repair",
    "repeat_teacher_mask_adapter",
    "threshold_relaxation",
    "manual_listening_required",
    "per_track_hardcode",
    "fallback_as_success",
}
REQUIRED_STEMS = ("vocal", "minus")
MANIFEST_NAME = "manifest.json"
PLAN_NAME = "plan.json"
PROGRESS_NAME = "progress.jsonl"
CANDIDATES_NAME = "candidates.jsonl"
GATES_NAME = "gates.jsonl"
REJECTED_NAME = "rejected.jsonl"
ACCEPTED_NAME = "accepted.jsonl"
SUMMARY_NAME = "summary.md"
START_TIME = time.monotonic()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _ensure_output_inside_rag(*, artifact_rag_root: Path, output_dir: Path) -> tuple[Path, Path]:
    root = _resolved(artifact_rag_root)
    output = _resolved(output_dir)
    if output != root and root not in output.parents:
        raise ValueError("output_dir must be inside artifact_rag_root")
    try:
        validate_artifact_relative_path(output.relative_to(root).as_posix())
    except ValueError as exc:
        raise ValueError("output_dir must use short Windows-safe relative path inside artifact_rag_root") from exc
    return root, output


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _flatten_plan_tokens(plan: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key in ("route_id", "primary_route", "strategy", "mode"):
        if plan.get(key):
            tokens.append(_normalize_token(plan.get(key)))
    for key in ("candidate_families", "forbidden_primary_directions", "route_matrix", "routes"):
        raw = plan.get(key)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    tokens.extend(_normalize_token(value) for value in item.values())
                else:
                    tokens.append(_normalize_token(item))
        elif isinstance(raw, dict):
            tokens.extend(_normalize_token(value) for value in raw.values())
        elif raw:
            tokens.append(_normalize_token(raw))
    return [token for token in tokens if token]


def _reject_forbidden_directions(plan: dict[str, Any]) -> None:
    for token in _flatten_plan_tokens(plan):
        if token in FORBIDDEN_PRIMARY_DIRECTIONS:
            raise ValueError(f"forbidden primary direction: {token}")


def _plan_families(plan: dict[str, Any]) -> list[str]:
    raw = plan.get("candidate_families") or []
    if not isinstance(raw, list):
        raise ValueError("candidate_families must be a list")
    families = [_normalize_token(item) for item in raw if _normalize_token(item)]
    return families


def _progress_row(event: str, **fields: Any) -> dict[str, Any]:
    row = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_dsp_scoring",
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "existing_candidate_pool_scoring_only",
        "track_alias": str(fields.get("track_alias") or ""),
        "stem": str(fields.get("stem") or ""),
        "candidate_alias": str(fields.get("candidate_alias") or ""),
        "phase": event,
        "delivery_allowed": False,
        "release_ready": False,
    }
    row.update(fields)
    return row


def append_progress(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_progress_row(event, **fields), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _validate_alias(value: Any, *, label: str) -> str:
    alias = str(value or "").strip()
    try:
        validate_artifact_segment(alias)
    except ValueError as exc:
        raise ValueError(f"unsafe artifact alias: {label}={alias}") from exc
    return alias


def _validate_candidate_row(row: dict[str, Any], *, allowed_families: set[str]) -> dict[str, Any]:
    candidate = dict(row)
    family = _normalize_token(candidate.get("family") or candidate.get("source_candidate_family"))
    if family in FORBIDDEN_PRIMARY_DIRECTIONS:
        raise ValueError(f"forbidden primary direction: {family}")
    if allowed_families and family and family not in allowed_families:
        candidate["status"] = "skipped"
        candidate["skip_reason"] = "family_not_in_rag_plan"
    track_alias = _validate_alias(candidate.get("track_alias"), label="track_alias")
    candidate_alias = _validate_alias(candidate.get("candidate_alias") or candidate.get("id"), label="candidate_alias")
    stem = _normalize_token(candidate.get("stem") or candidate.get("stem_type"))
    if stem not in REQUIRED_STEMS:
        candidate["status"] = "skipped"
        candidate["skip_reason"] = "stem_not_required_for_vocal_minus_pair"
    short_path = str(candidate.get("short_audio_path") or candidate.get("artifact_relative_path") or "")
    if short_path:
        try:
            validate_artifact_relative_path(short_path)
        except ValueError as exc:
            raise ValueError(f"unsafe artifact alias: short_audio_path={short_path}") from exc
    candidate.update(
        {
            "schema": "aris_refinement_candidate_row.v1",
            "track_alias": track_alias,
            "candidate_alias": candidate_alias,
            "stem": stem,
            "family": family,
            "delivery_allowed": False,
            "release_ready": False,
        }
    )
    return candidate


def _candidate_failed_checks(candidate: dict[str, Any]) -> list[str]:
    checks = candidate.get("failed_checks") or candidate.get("failed_check_names") or []
    if isinstance(checks, str):
        checks = [checks]
    return sorted({str(check) for check in checks if str(check)})


def _truthy_contract_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _normalize_token(value) in {"1", "true", "yes", "y", "used"}


def _candidate_has_forbidden_fallback(candidate: dict[str, Any]) -> bool:
    return _truthy_contract_flag(candidate.get("fallback_used")) or _truthy_contract_flag(
        candidate.get("fallback_as_success")
    )


def _candidate_has_objective_gate(candidate: dict[str, Any]) -> bool:
    return _normalize_token(candidate.get("reference_gate")) == "passed" or _normalize_token(candidate.get("no_reference_gate")) == "passed"


def _candidate_passed(candidate: dict[str, Any]) -> bool:
    status = _normalize_token(candidate.get("candidate_status") or candidate.get("status"))
    if status not in {"passed", "accepted"}:
        return False
    if _candidate_failed_checks(candidate):
        return False
    if _candidate_has_forbidden_fallback(candidate):
        return False
    if _normalize_token(candidate.get("pair_consistency_gate") or "failed") != "passed":
        return False
    return _candidate_has_objective_gate(candidate)


def _candidate_blockers(candidate: dict[str, Any], stem: str) -> list[str]:
    failed_checks = _candidate_failed_checks(candidate)
    if failed_checks:
        return failed_checks
    blockers: list[str] = []
    status = _normalize_token(candidate.get("candidate_status") or candidate.get("status"))
    if status not in {"passed", "accepted"}:
        blockers.append(f"{stem}_candidate_not_accepted")
    if _candidate_has_forbidden_fallback(candidate):
        blockers.append("fallback_used_not_allowed")
    if _normalize_token(candidate.get("pair_consistency_gate") or "failed") != "passed":
        blockers.append("pair_consistency_gate_failed")
    if not _candidate_has_objective_gate(candidate):
        blockers.append("objective_gate_missing")
    return sorted(set(blockers))


def _select_stem_candidate(candidates: Sequence[dict[str, Any]], stem: str, family_rank: dict[str, int]) -> dict[str, Any] | None:
    stem_candidates = [candidate for candidate in candidates if candidate.get("stem") == stem and candidate.get("status") != "skipped"]
    passed = [candidate for candidate in stem_candidates if _candidate_passed(candidate)]
    pool = passed or stem_candidates
    return min(
        pool,
        key=lambda candidate: (
            0 if _candidate_passed(candidate) else 1,
            family_rank.get(str(candidate.get("family") or ""), 999),
            str(candidate.get("candidate_alias") or ""),
        ),
        default=None,
    )


def _combined_gate(stem_rows: Sequence[dict[str, Any]], key: str, *, accepted_values: set[str]) -> str:
    values = [_normalize_token(row.get(key)) for row in stem_rows if row.get(key) is not None]
    if values and all(value in accepted_values for value in values):
        if any(value == "passed" for value in values):
            return "passed"
        return sorted(values)[0]
    if key == "reference_gate":
        return "not_available"
    if key == "no_reference_gate":
        return "not_required"
    return "failed"


def _track_failed_checks(selected: dict[str, dict[str, Any] | None]) -> list[str]:
    failed: set[str] = set()
    for stem, row in selected.items():
        if row is None:
            failed.add(f"{stem}_candidate_missing")
            continue
        failed.update(_candidate_blockers(row, stem))
    return sorted(failed)


def _accepted_row(track_alias: str, selected_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "aris_refinement_acceptance_row.v1",
        "track_alias": track_alias,
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_ready": False,
        "accepted_stems": list(REQUIRED_STEMS),
        "candidate_aliases": {row["stem"]: row["candidate_alias"] for row in selected_rows},
        "short_audio_paths": {
            row["stem"]: str(row.get("short_audio_path") or row.get("artifact_relative_path") or "")
            for row in selected_rows
        },
        "objective_evidence": {
            "reference_gate": _combined_gate(selected_rows, "reference_gate", accepted_values={"passed", "not_available"}),
            "no_reference_gate": _combined_gate(selected_rows, "no_reference_gate", accepted_values={"passed", "not_required"}),
            "pair_consistency_gate": "passed",
        },
    }


def _rejected_row(track_alias: str, selected: dict[str, dict[str, Any] | None], failed_checks: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": "aris_refinement_rejection_row.v1",
        "track_alias": track_alias,
        "delivery_status": "rejected",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_ready": False,
        "fallback_used": False,
        "dominant_failure_reason": str(next(iter(failed_checks), "candidate_pair_not_accepted")),
        "failed_checks": sorted(set(str(check) for check in failed_checks)),
        "selected_candidates": {
            stem: None
            if row is None
            else {
                "candidate_alias": row.get("candidate_alias"),
                "family": row.get("family"),
                "candidate_status": row.get("candidate_status") or row.get("status"),
                "fallback_used": _truthy_contract_flag(row.get("fallback_used")),
                "failed_checks": _candidate_failed_checks(row),
            }
            for stem, row in selected.items()
        },
    }


def run_aris_refinement_matrix(
    *,
    artifact_rag_root: Path,
    planning_json: Path,
    candidate_pool_jsonl: Path,
    output_dir: Path,
    target_stems: Sequence[str] = REQUIRED_STEMS,
    command_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    root, output = _ensure_output_inside_rag(artifact_rag_root=artifact_rag_root, output_dir=output_dir)
    stems = tuple(_normalize_token(stem) for stem in target_stems)
    if stems != REQUIRED_STEMS:
        raise ValueError("run_aris_refinement_matrix currently requires target_stems vocal,minus")
    plan = read_json(planning_json)
    _reject_forbidden_directions(plan)
    families = _plan_families(plan)
    family_rank = {family: index for index, family in enumerate(families)}
    allowed_families = set(families)
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    append_progress(
        progress_path,
        "run_start",
        planning_json=str(planning_json),
        candidate_pool_jsonl=str(candidate_pool_jsonl),
        output_dir=str(output),
    )

    candidates = [_validate_candidate_row(row, allowed_families=allowed_families) for row in iter_jsonl(candidate_pool_jsonl)]
    candidates_by_track: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        candidates_by_track.setdefault(str(candidate["track_alias"]), []).append(candidate)

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for track_alias in sorted(candidates_by_track):
        append_progress(progress_path, "track_start", track_alias=track_alias)
        track_candidates = candidates_by_track[track_alias]
        selected = {
            stem: _select_stem_candidate(track_candidates, stem, family_rank)
            for stem in REQUIRED_STEMS
        }
        failed_checks = _track_failed_checks(selected)
        selected_rows = [row for row in selected.values() if row is not None]
        reference_gate = _combined_gate(selected_rows, "reference_gate", accepted_values={"passed", "not_available"})
        no_reference_gate = _combined_gate(selected_rows, "no_reference_gate", accepted_values={"passed", "not_required"})
        pair_consistency_gate = "passed" if not failed_checks else "failed"
        gate_row = {
            "schema": "aris_refinement_gate_row.v1",
            "track_alias": track_alias,
            "status": "accepted" if not failed_checks else "rejected",
            "reference_gate": reference_gate,
            "no_reference_gate": no_reference_gate,
            "pair_consistency_gate": pair_consistency_gate,
            "failed_checks": failed_checks,
            "delivery_allowed": False,
            "release_ready": False,
        }
        gate_rows.append(gate_row)
        append_progress(
            progress_path,
            "gate_finish",
            track_alias=track_alias,
            status=gate_row["status"],
            failed_checks=failed_checks,
        )
        if failed_checks:
            rejected_rows.append(_rejected_row(track_alias, selected, failed_checks))
        else:
            accepted_rows.append(_accepted_row(track_alias, selected_rows))
        append_progress(progress_path, "track_finish", track_alias=track_alias, status=gate_row["status"])

    accepted_path = output / ACCEPTED_NAME
    rejected_path = output / REJECTED_NAME
    candidates_path = output / CANDIDATES_NAME
    gates_path = output / GATES_NAME
    plan_path = output / PLAN_NAME
    summary_path = output / SUMMARY_NAME
    write_json(plan_path, plan)
    write_jsonl(candidates_path, candidates)
    write_jsonl(gates_path, gate_rows)
    write_jsonl(accepted_path, accepted_rows)
    write_jsonl(rejected_path, rejected_rows)
    overall_accepted = bool(accepted_rows) and not rejected_rows and len(accepted_rows) == len(candidates_by_track)
    manifest = {
        "schema": "aris_refinement_matrix_manifest.v1",
        "manifest_type": "aris_refinement_matrix",
        "status": "accepted" if overall_accepted else "blocked",
        "artifact_rag_root": str(root),
        "output_dir": str(output),
        "planning_json": str(planning_json),
        "candidate_pool_jsonl": str(candidate_pool_jsonl),
        "copied_plan_json": str(plan_path),
        "progress_jsonl": str(progress_path),
        "candidates_jsonl": str(candidates_path),
        "gates_jsonl": str(gates_path),
        "accepted_jsonl": str(accepted_path),
        "rejected_jsonl": str(rejected_path),
        "summary_md": str(summary_path),
        "target_stems": list(REQUIRED_STEMS),
        "candidate_families": families,
        "track_count": len(candidates_by_track),
        "candidate_count": len(candidates),
        "accepted_count": len(accepted_rows),
        "rejected_count": len(rejected_rows),
        "delivery_status": "accepted" if overall_accepted else "blocked",
        "delivery_allowed": overall_accepted,
        "presentation_status": "accepted_for_listen_pack" if overall_accepted else "blocked_from_listen_pack",
        "release_ready": False,
        "can_publish_as_result": False,
        "fallback_used": False,
        "thresholds_relaxed": False,
        "forbidden_primary_directions": sorted(FORBIDDEN_PRIMARY_DIRECTIONS),
        "command_args": list(command_args or []),
        "contract": {
            "rag_controlled_outputs_only": True,
            "best_of_bad_candidate_delivery_forbidden": True,
            "rejected_candidates_are_diagnostic_only": True,
            "qwen_controller_forbidden": True,
            "diffusion_generative_repair_forbidden": True,
            "repeat_teacher_mask_adapter_forbidden": True,
        },
    }
    write_json(output / MANIFEST_NAME, manifest)
    summary_path.write_text(
        "\n".join(
            [
                "# ARIS Refinement Matrix",
                "",
                f"Status: {manifest['status']}",
                f"Tracks: {manifest['track_count']}",
                f"Accepted: {manifest['accepted_count']}",
                f"Rejected: {manifest['rejected_count']}",
                "Rejected rows are diagnostic-only and blocked from listen packs.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    append_progress(
        progress_path,
        "run_finish",
        status=manifest["status"],
        accepted_count=manifest["accepted_count"],
        rejected_count=manifest["rejected_count"],
        manifest_path=str(output / MANIFEST_NAME),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ARIS failure-conditioned vocal/minus refinement matrix.")
    parser.add_argument("--artifact-rag-root", type=Path, required=True)
    parser.add_argument("--planning-json", type=Path, required=True)
    parser.add_argument("--candidate-pool-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-stem", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_aris_refinement_matrix.py",
        entrypoint_path="/app/tools/run_aris_refinement_matrix.py",
    )
    args = parse_args(argv)
    manifest = run_aris_refinement_matrix(
        artifact_rag_root=args.artifact_rag_root,
        planning_json=args.planning_json,
        candidate_pool_jsonl=args.candidate_pool_jsonl,
        output_dir=args.output_dir,
        target_stems=args.target_stem or list(REQUIRED_STEMS),
        command_args=list(argv or sys.argv[1:]),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
