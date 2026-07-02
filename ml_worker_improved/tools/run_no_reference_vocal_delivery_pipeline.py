from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_vocal_listen_preview_manifest import (  # noqa: E402
    SAFETY_FLAGS,
    build_manifest as build_vocal_listen_preview_manifest,
    build_summary as build_vocal_listen_preview_summary,
    write_json as write_contract_json,
)
from tools.run_no_reference_candidate_selector import write_outputs as write_selector_outputs  # noqa: E402


PIPELINE_MANIFEST_NAME = "no_reference_vocal_delivery_pipeline_manifest.json"
PIPELINE_SUMMARY_NAME = "no_reference_vocal_delivery_pipeline_summary.md"
ACCEPTED_SELECTOR_DECISIONS = {"accept_candidate", "select_alternative_candidate"}
STEM_MIN_WINDOW_COUNT = 5

START_TIME = time.monotonic()


def log_event(phase: str, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "phase": phase,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_inference_skipped": True,
        "gpu_inference_skipped_reason": "no_reference_vocal_delivery_uses_existing_vocal_judge_rows",
    }
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def source_paths(paths: Iterable[Path]) -> list[str]:
    return [str(path) for path in paths]


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value is None:
        return []
    text = str(value)
    return [text] if text else []


def stable_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def vocal_path_from_row(row: dict[str, Any]) -> str:
    return str(row.get("selected_vocal_path") or row.get("vocal_path") or "")


def candidate_id_from_row(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or "")


def row_float(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return float(value)
    return 0.0


def row_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key))
    except (TypeError, ValueError):
        return default


def vocal_score_from_row(row: dict[str, Any]) -> float:
    return row_float(row, "vocal_score")


def confidence_from_row(row: dict[str, Any]) -> float:
    return row_float(row, "confidence")


def vocal_acceptance_decision(row: dict[str, Any]) -> str:
    return str(row.get("vocal_decision") or "")


def is_vocal_accepted_row(row: dict[str, Any]) -> bool:
    return (
        str(row.get("score_mode") or "") == "vocal"
        and row.get("vocal_score") not in (None, "")
        and vocal_acceptance_decision(row) == "accept_candidate"
        and row.get("vocal_blockers") is not None
        and not as_list(row.get("vocal_blockers"))
        and row_int(row, "window_count") >= STEM_MIN_WINDOW_COUNT
    )


def accepted_vocal_rows(rows: list[dict[str, Any]], base_candidate_id: str | None = None) -> list[dict[str, Any]]:
    accepted = [row for row in rows if is_vocal_accepted_row(row)]
    if base_candidate_id:
        accepted = [row for row in accepted if candidate_id_from_row(row) == base_candidate_id]
    return accepted


def _matrix_row_identifiers(row: dict[str, Any]) -> set[str]:
    identifiers = {
        str(row.get("canonical_candidate_id") or ""),
        str(row.get("candidate_id") or ""),
    }
    aliases = row.get("aliases")
    if isinstance(aliases, list):
        identifiers.update(str(alias) for alias in aliases if str(alias))
    vocal_judge = row.get("vocal_judge")
    if isinstance(vocal_judge, dict):
        identifiers.add(str(vocal_judge.get("candidate_id") or ""))
    return {identifier for identifier in identifiers if identifier}


def choose_base_candidate_id_from_score_matrix(
    *,
    matrix_json: Path,
    vocal_rows: list[dict[str, Any]],
) -> str | None:
    payload = json.loads(matrix_json.read_text(encoding="utf-8-sig"))
    rows = payload.get("rows") if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else []
    labels = {candidate_id_from_row(row): row for row in vocal_rows if candidate_id_from_row(row)}
    actionable_next_steps = {
        "candidate_ready_for_listen_preview_contract",
        "run_selector_and_listen_preview_contract",
        "accepted_for_listen_pack_not_release",
    }
    ranked_rows = sorted(
        [row for row in rows if isinstance(row, dict)],
        key=lambda row: (
            int(row.get("rank") or 999999),
            -float(row.get("selection_priority") or 0.0),
            str(row.get("canonical_candidate_id") or ""),
        ),
    )
    for row in ranked_rows:
        if str(row.get("next_action") or "") not in actionable_next_steps:
            continue
        canonical_candidate_id = str(row.get("canonical_candidate_id") or "")
        if canonical_candidate_id in labels:
            return canonical_candidate_id
        vocal_judge = row.get("vocal_judge")
        if isinstance(vocal_judge, dict):
            matrix_candidate_id = str(vocal_judge.get("candidate_id") or "")
            if matrix_candidate_id in labels:
                return matrix_candidate_id
        row_identifiers = _matrix_row_identifiers(row)
        matching_labels = [label for label in labels if label and label in row_identifiers]
        if matching_labels:
            return sorted(matching_labels, key=lambda label: (-len(label), label))[0]
    return None


def missing_audio_inputs(row: dict[str, Any]) -> list[str]:
    paths = [
        Path(str(row.get("mix_path") or row.get("source_audio_path") or "")),
        Path(vocal_path_from_row(row)),
    ]
    return [str(path) for path in paths if not str(path) or not path.exists()]


def ranked_vocal_candidate(row: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "candidate_id": candidate_id_from_row(row),
        "vocal_score": vocal_score_from_row(row),
        "overall_no_reference_score": row_float(row, "overall_no_reference_score"),
        "confidence": confidence_from_row(row),
        "decision": str(row.get("decision") or ""),
        "reasons": as_list(row.get("reasons")),
    }


def select_vocal_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        track_id = str(row.get("track_id") or "")
        if not track_id:
            raise ValueError(f"Vocal judge row is missing track_id: {row}")
        grouped.setdefault(track_id, []).append(row)

    selected_rows: list[dict[str, Any]] = []
    for track_id in sorted(grouped):
        candidates = grouped[track_id]
        ordered = sorted(
            candidates,
            key=lambda row: (-vocal_score_from_row(row), -confidence_from_row(row), candidate_id_from_row(row)),
        )
        top = ordered[0]
        decision = "accept_candidate" if len(ordered) == 1 else "select_alternative_candidate"
        reasons = stable_unique(["selected_by_vocal_score"] + as_list(top.get("reasons")))
        selected_rows.append(
            {
                "track_id": track_id,
                "decision": decision,
                "selected_candidate_id": candidate_id_from_row(top),
                "selected_candidate": top,
                "candidate_count": len(ordered),
                "accepted_candidate_count": len(ordered),
                "ranked_candidates": [ranked_vocal_candidate(row, index + 1) for index, row in enumerate(ordered)],
                "confidence": confidence_from_row(top),
                "reasons": reasons,
                "segment_blend_plan": None,
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
                "score_mode": "vocal",
                "selection_metric": "vocal_score",
            }
        )

    decision_counts = Counter(str(row.get("decision") or "") for row in selected_rows)
    reasons = Counter(reason for row in selected_rows for reason in as_list(row.get("reasons")))
    summary = {
        "total_tracks": len(selected_rows),
        "candidates_judged": sum(int(row.get("candidate_count") or 0) for row in selected_rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "accepted_candidates": decision_counts.get("accept_candidate", 0),
        "selected_alternatives": decision_counts.get("select_alternative_candidate", 0),
        "segment_blend_needed": 0,
        "blocked_hard_case": 0,
        "usable_without_stems": decision_counts.get("accept_candidate", 0) + decision_counts.get("select_alternative_candidate", 0),
        "top_failure_reasons": dict(reasons.most_common(10)),
        "safety": dict(SAFETY_FLAGS),
        "score_mode": "vocal",
        "selection_metric": "vocal_score",
    }
    return {"rows": selected_rows, "summary": summary}


def candidate_manifest_from_row(*, row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    vocal_path = vocal_path_from_row(row)
    candidate_id = candidate_id_from_row(row)
    track_id = str(row.get("track_id") or Path(vocal_path).stem or candidate_id)
    render_status = str(row.get("render_status") or "existing_vocal_candidate_proxy_pass")
    candidate_row = {
        **row,
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": str(row.get("mix_path") or row.get("source_audio_path") or ""),
        "source_audio_path": str(row.get("source_audio_path") or row.get("mix_path") or ""),
        "vocal_path": vocal_path,
        "selected_vocal_path": vocal_path,
        "instrumental_path": str(row.get("instrumental_path") or row.get("minus_path") or ""),
        "minus_path": str(row.get("minus_path") or row.get("instrumental_path") or ""),
        "selected_minus_path": str(row.get("selected_minus_path") or row.get("minus_path") or row.get("instrumental_path") or ""),
        "candidate_family": str(row.get("candidate_family") or "existing_vocal_proxy_pass"),
        "candidate_origin": str(row.get("candidate_origin") or "existing_vocal_mode_judge_proxy_pass"),
        "base_candidate_id": candidate_id,
        "render_status": render_status,
        "final_decision": "existing_vocal_candidate",
        "requires_post_render_vocal_judge": True,
        "requires_dense_post_reconciliation_judge": False,
        "needs_pair_reconciliation": False,
        "segment_blend_needed": False,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "delivery_allowed": False,
        "delivery_status": "blocked",
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "is_complete_vocal_candidate": bool(vocal_path),
    }
    candidate_row_jsonl = output_dir / "existing_vocal_candidate_row.jsonl"
    write_jsonl(candidate_row_jsonl, [candidate_row])
    return {
        "manifest_type": "no_reference_vocal_existing_candidate",
        "track_id": track_id,
        "candidate_id": candidate_id,
        "base_candidate_id": candidate_id,
        "render_status": render_status,
        "dominant_failure_reason": "",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_post_render_vocal_judge": True,
        "requires_dense_post_reconciliation_judge": False,
        "rendered_vocal_candidate_path": vocal_path,
        "candidate_row_jsonl": str(candidate_row_jsonl),
        "candidate_row": candidate_row,
        "safety_flags": dict(SAFETY_FLAGS),
        "audio_copied": False,
        "audio_exported": False,
        "audio_exported_as_result": False,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
    }


def blocked_pipeline_manifest(
    *,
    output_dir: Path,
    vocal_judge_jsonl: Path,
    reason: str,
    base_candidate_id: str,
) -> dict[str, Any]:
    return {
        "manifest_type": "no_reference_vocal_delivery_pipeline",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": reason,
        "stem_type": "vocal",
        "base_candidate_id": base_candidate_id,
        "source_vocal_judge_jsonl": str(vocal_judge_jsonl),
        "candidate_manifest_path": str(output_dir / "vocal_candidate_manifest.json"),
        "selector_jsonl": str(output_dir / "no_reference_candidate_selector" / "no_reference_candidate_selection.jsonl"),
        "final_listen_preview_manifest": "",
        "accepted_count": 0,
        "skipped_count": 1,
        "score_modes_used": ["vocal"],
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": dict(SAFETY_FLAGS),
    }


def accepted_pipeline_manifest(
    *,
    output_dir: Path,
    vocal_judge_jsonl: Path,
    base_candidate_id: str,
    listen_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_type": "no_reference_vocal_delivery_pipeline",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "aggregate_machine_gate_required_not_for_listen_pack",
        "release_status": "not_release_ready",
        "dominant_failure_reason": "requires_staged_aggregate_validation_and_machine_gate",
        "stem_type": "vocal",
        "base_candidate_id": base_candidate_id,
        "listen_source_promotable": True,
        "source_listen_preview_delivery_status": str(listen_manifest.get("delivery_status") or ""),
        "source_listen_preview_delivery_allowed": listen_manifest.get("delivery_allowed") is True,
        "source_listen_preview_presentation_status": str(listen_manifest.get("presentation_status") or ""),
        "source_listen_preview_release_status": str(listen_manifest.get("release_status") or ""),
        "requires_staged_aggregate_machine_gate": True,
        "final_delivery_blockers": ["requires_staged_aggregate_validation_and_machine_gate"],
        "source_vocal_judge_jsonl": str(vocal_judge_jsonl),
        "candidate_manifest_path": str(output_dir / "vocal_candidate_manifest.json"),
        "selector_jsonl": str(output_dir / "no_reference_candidate_selector" / "no_reference_candidate_selection.jsonl"),
        "final_listen_preview_manifest": str(output_dir / "no_reference_vocal_listen_preview_manifest.json"),
        "accepted_count": int(listen_manifest.get("accepted_count") or 0),
        "skipped_count": int(listen_manifest.get("skipped_count") or 0),
        "selected_vocal_path": listen_manifest.get("selected_vocal_path") or "",
        "score_modes_used": ["vocal"],
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": dict(SAFETY_FLAGS),
    }


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Vocal Delivery Pipeline Summary",
        "",
        f"- delivery_status: {manifest.get('delivery_status')}",
        f"- delivery_allowed: {str(manifest.get('delivery_allowed') is True).lower()}",
        f"- presentation_status: {manifest.get('presentation_status')}",
        f"- release_status: {manifest.get('release_status')}",
        f"- stem_type: {manifest.get('stem_type')}",
        f"- accepted_count: {manifest.get('accepted_count')}",
        f"- skipped_count: {manifest.get('skipped_count')}",
        f"- dominant_failure_reason: {manifest.get('dominant_failure_reason')}",
        f"- final_listen_preview_manifest: {manifest.get('final_listen_preview_manifest')}",
        "",
        "## Safety",
        "",
    ]
    safety = manifest.get("safety") if isinstance(manifest.get("safety"), dict) else {}
    for key in sorted(SAFETY_FLAGS):
        lines.append(f"- {key}: {str(safety.get(key) is True).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pipeline_outputs(output_dir: Path, manifest: dict[str, Any]) -> None:
    write_contract_json(output_dir / PIPELINE_MANIFEST_NAME, manifest)
    write_summary(output_dir / PIPELINE_SUMMARY_NAME, manifest)


def annotate_base_selection(
    manifest: dict[str, Any],
    *,
    base_selection_source: str,
    candidate_score_matrix_json: Path | None,
) -> dict[str, Any]:
    manifest["base_selection_source"] = base_selection_source
    manifest["candidate_score_matrix_json"] = str(candidate_score_matrix_json or "")
    return manifest


def run_vocal_delivery_pipeline(
    *,
    vocal_judge_jsonl: Path,
    output_dir: Path,
    base_candidate_id: str | None,
    candidate_score_matrix_json: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_event("prepare_vocal_candidates", "start", vocal_judge_jsonl=str(vocal_judge_jsonl), output_dir=str(output_dir))
    rows = list(iter_jsonl(vocal_judge_jsonl))
    base_selection_source = "explicit_base_candidate_id" if base_candidate_id else "vocal_score_sort"
    matrix_base_candidate_id = None
    if base_candidate_id is None and candidate_score_matrix_json is not None:
        matrix_base_candidate_id = choose_base_candidate_id_from_score_matrix(
            matrix_json=candidate_score_matrix_json,
            vocal_rows=rows,
        )
        if matrix_base_candidate_id:
            base_selection_source = "candidate_score_matrix"
    selected_base_candidate_id = base_candidate_id or matrix_base_candidate_id
    log_event(
        "base_selection",
        "end",
        base_candidate_id=str(selected_base_candidate_id or ""),
        base_selection_source=base_selection_source,
        candidate_score_matrix_json=str(candidate_score_matrix_json or ""),
    )
    accepted_rows = accepted_vocal_rows(rows, base_candidate_id=selected_base_candidate_id)
    if not accepted_rows:
        manifest = blocked_pipeline_manifest(
            output_dir=output_dir,
            vocal_judge_jsonl=vocal_judge_jsonl,
            reason="no_accepted_vocal_candidate",
            base_candidate_id=str(selected_base_candidate_id or ""),
        )
        annotate_base_selection(
            manifest,
            base_selection_source=base_selection_source,
            candidate_score_matrix_json=candidate_score_matrix_json,
        )
        write_pipeline_outputs(output_dir, manifest)
        return manifest

    candidate_rows_jsonl = output_dir / "no_reference_vocal_delivery_candidates.jsonl"
    write_jsonl(candidate_rows_jsonl, accepted_rows)
    selected_input_row = sorted(
        accepted_rows,
        key=lambda row: (
            -vocal_score_from_row(row),
            -float(row.get("confidence") or 0.0),
            candidate_id_from_row(row),
        ),
    )[0]
    missing_inputs = missing_audio_inputs(selected_input_row)
    if missing_inputs:
        manifest = blocked_pipeline_manifest(
            output_dir=output_dir,
            vocal_judge_jsonl=vocal_judge_jsonl,
            reason="missing_audio_path_for_vocal_delivery",
            base_candidate_id=candidate_id_from_row(selected_input_row),
        )
        manifest["missing_audio_inputs"] = missing_inputs
        annotate_base_selection(
            manifest,
            base_selection_source=base_selection_source,
            candidate_score_matrix_json=candidate_score_matrix_json,
        )
        write_pipeline_outputs(output_dir, manifest)
        return manifest

    selector_dir = output_dir / "no_reference_candidate_selector"
    selector_result = select_vocal_candidates(accepted_rows)
    write_selector_outputs(selector_result, selector_dir)
    selector_rows = selector_result.get("rows") if isinstance(selector_result.get("rows"), list) else []
    selector_row = selector_rows[0] if selector_rows else {}
    if str(selector_row.get("decision") or "") not in ACCEPTED_SELECTOR_DECISIONS:
        manifest = blocked_pipeline_manifest(
            output_dir=output_dir,
            vocal_judge_jsonl=vocal_judge_jsonl,
            reason="selector_not_accepted",
            base_candidate_id=candidate_id_from_row(selected_input_row),
        )
        annotate_base_selection(
            manifest,
            base_selection_source=base_selection_source,
            candidate_score_matrix_json=candidate_score_matrix_json,
        )
        write_pipeline_outputs(output_dir, manifest)
        return manifest

    selected_candidate = selector_row.get("selected_candidate") if isinstance(selector_row.get("selected_candidate"), dict) else selected_input_row
    render_manifest = candidate_manifest_from_row(row=selected_candidate, output_dir=output_dir)
    candidate_manifest_json = output_dir / "vocal_candidate_manifest.json"
    write_contract_json(candidate_manifest_json, render_manifest)

    listen_manifest_json = output_dir / "no_reference_vocal_listen_preview_manifest.json"
    listen_summary_md = output_dir / "no_reference_vocal_listen_preview_summary.md"
    listen_manifest = build_vocal_listen_preview_manifest(
        candidate_manifest_json=candidate_manifest_json,
        vocal_judge_jsonl=vocal_judge_jsonl,
        selector_jsonl=selector_dir / "no_reference_candidate_selection.jsonl",
    )
    write_contract_json(listen_manifest_json, listen_manifest)
    listen_summary_md.write_text(build_vocal_listen_preview_summary(listen_manifest), encoding="utf-8")

    if listen_manifest.get("delivery_allowed") is not True:
        manifest = blocked_pipeline_manifest(
            output_dir=output_dir,
            vocal_judge_jsonl=vocal_judge_jsonl,
            reason=str(listen_manifest.get("dominant_failure_reason") or "vocal_listen_preview_not_accepted"),
            base_candidate_id=candidate_id_from_row(selected_candidate),
        )
        manifest["final_listen_preview_manifest"] = str(listen_manifest_json)
        manifest["skipped_count"] = int(listen_manifest.get("skipped_count") or 1)
        annotate_base_selection(
            manifest,
            base_selection_source=base_selection_source,
            candidate_score_matrix_json=candidate_score_matrix_json,
        )
        write_pipeline_outputs(output_dir, manifest)
        return manifest

    manifest = accepted_pipeline_manifest(
        output_dir=output_dir,
        vocal_judge_jsonl=vocal_judge_jsonl,
        base_candidate_id=candidate_id_from_row(selected_candidate),
        listen_manifest=listen_manifest,
    )
    annotate_base_selection(
        manifest,
        base_selection_source=base_selection_source,
        candidate_score_matrix_json=candidate_score_matrix_json,
    )
    write_pipeline_outputs(output_dir, manifest)
    log_event("vocal_listen_preview_manifest", "end", listen_source_promotable=True, output_manifest=str(listen_manifest_json))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the no-reference vocal-only delivery pipeline from vocal judge rows.")
    parser.add_argument("--vocal-judge-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-candidate-id")
    parser.add_argument("--candidate-score-matrix-json", type=Path)
    args = parser.parse_args(argv)

    manifest = run_vocal_delivery_pipeline(
        vocal_judge_jsonl=args.vocal_judge_jsonl,
        output_dir=args.output_dir,
        base_candidate_id=str(args.base_candidate_id) if args.base_candidate_id else None,
        candidate_score_matrix_json=args.candidate_score_matrix_json,
    )
    print(
        json.dumps(
            {
                key: manifest.get(key)
                for key in (
                    "delivery_status",
                    "delivery_allowed",
                    "presentation_status",
                    "release_status",
                    "accepted_count",
                    "skipped_count",
                    "dominant_failure_reason",
                    "final_listen_preview_manifest",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest.get("delivery_allowed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
