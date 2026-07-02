from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime


SCHEMA = "minus_repair_outcome_training_view.v1"
MANIFEST_NAME = "manifest.json"
ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
START_TIME = time.monotonic()

ACCEPT_STATUSES = {"accepted", "release_ready", "passed"}
REJECT_STATUSES = {"rejected", "blocked", "failed", "blocked_for_listening"}
ROUTING_ONLY_STATUSES = {"passed_for_candidate_routing"}
MIN_SAFE_REPAIR_ROWS = 10
MIN_POSITIVE_REPAIR_ROWS = 3
MIN_NEGATIVE_REPAIR_ROWS = 3
DETAIL_DAMAGE_BLOCKERS = {
    "rms_dip_damage",
    "presence_dip_damage",
    "air_dip_damage",
    "rms_boost_damage",
    "presence_boost_damage",
    "air_boost_damage",
    "cumulative_rms_dip_damage",
    "cumulative_presence_dip_damage",
    "cumulative_air_dip_damage",
    "cumulative_rms_boost_damage",
    "cumulative_presence_boost_damage",
    "cumulative_air_boost_damage",
}
LEAKAGE_BLOCKERS = {
    "no_meaningful_minus_delta",
    "insufficient_vocal_residue_improvement",
    "severe_vocal_bleed_in_minus",
    "vocal_projection_proxy",
    "vocal_stem_corr_proxy",
    "full_mix_like_risk_proxy",
}
SAFETY_FLAGS = {
    "delivery_allowed": False,
    "can_publish_as_result": False,
    "presentation_status": "repair_outcome_view_not_for_listen_pack",
    "release_ready": False,
    "separator_training_started": False,
    "model_training_started": False,
    "model_registry_edited": False,
    "thresholds_relaxed": False,
    "runtime_gates_changed": False,
    "audio_written": False,
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected_json_object:{path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"expected_json_object:{path}:{line_number}")
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
        "route": "minus_repair_outcome_training_view",
        "acceleration_mode": "external_only_cpu_metadata_build",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "builder reads existing repair manifests and matrix rows only",
        **fields,
    }
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def stable_unique(items: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def blocker_labels(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    labels: list[str] = []
    for item in value:
        if isinstance(item, dict):
            labels.append(str(item.get("label") or ""))
        else:
            labels.append(str(item or ""))
    return stable_unique(labels)


def status_label(status: Any) -> int | None:
    normalized = str(status or "").strip().lower()
    if normalized in ACCEPT_STATUSES:
        return 1
    if normalized in REJECT_STATUSES or normalized.startswith(("reject", "block", "fail")):
        return 0
    return None


def has_metric_evidence(metrics: dict[str, Any]) -> bool:
    for value in metrics.values():
        if as_optional_float(value) is not None:
            return True
    return False


def unsafe_reasons(
    *,
    label: int | None,
    status: str,
    metrics: dict[str, Any],
    route_id: str,
    candidate_id: str,
) -> list[str]:
    reasons: list[str] = []
    if label is None:
        reasons.append("missing_repair_outcome_label")
    if str(status or "").strip().lower() in ROUTING_ONLY_STATUSES:
        reasons.append("routing_only_status_not_repair_outcome_label")
    if not route_id:
        reasons.append("missing_route_id")
    if not candidate_id:
        reasons.append("missing_candidate_id")
    if not has_metric_evidence(metrics):
        reasons.append("missing_repair_metric_evidence")
    return reasons


def metric_bundle(
    *,
    score: Any = None,
    avg_improve: Any = None,
    peak_improve: Any = None,
    worst_rms: Any = None,
    max_rms: Any = None,
    worst_presence: Any = None,
    max_presence: Any = None,
    worst_air: Any = None,
    max_air: Any = None,
    focus_improve: Any = None,
    focus_peak: Any = None,
    focus_rms: Any = None,
) -> dict[str, float | None]:
    return {
        "score": as_optional_float(score),
        "average_vocal_improvement_db": as_optional_float(avg_improve),
        "peak_corr_improvement": as_optional_float(peak_improve),
        "worst_rms_dip_db": as_optional_float(worst_rms),
        "max_rms_boost_db": as_optional_float(max_rms),
        "worst_presence_dip_db": as_optional_float(worst_presence),
        "max_presence_boost_db": as_optional_float(max_presence),
        "worst_air_dip_db": as_optional_float(worst_air),
        "max_air_boost_db": as_optional_float(max_air),
        "focus_average_vocal_improvement_db": as_optional_float(focus_improve),
        "focus_peak_corr_improvement": as_optional_float(focus_peak),
        "focus_worst_rms_dip_db": as_optional_float(focus_rms),
    }


def merge_metric_bundles(*bundles: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for bundle in bundles:
        for key, value in bundle.items():
            merged[key] = as_optional_float(value)
    return merged


def route_family_from_matrix_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "fswp" in parts:
        return "focus_strength_sweep"
    if "gblend" in parts:
        return "guide_blend_sweep"
    if "hnorm" in parts:
        return "level_normalized_historical_dedicated_minus_matrix"
    if "hist" in parts:
        return "historical_dedicated_minus_matrix"
    return "minus_repair_matrix"


def source_id(path: Path) -> str:
    parts = list(path.parts)
    tail = parts[-4:] if len(parts) >= 4 else parts
    raw = "_".join(Path(part).stem for part in tail if part)
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
    return safe[:48] or stable_hash(str(path))[:12]


def build_common_row(
    *,
    source_kind: str,
    source_path: Path,
    source_index: int,
    track_id: str,
    source_candidate_id: str,
    repair_candidate_id: str,
    route_id: str,
    route_family: str,
    status: str,
    blockers: Sequence[str],
    metrics: dict[str, Any],
    candidate_audio_path: str = "",
    parameters: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label = status_label(status)
    reasons = unsafe_reasons(
        label=label,
        status=status,
        metrics=metrics,
        route_id=route_id,
        candidate_id=repair_candidate_id,
    )
    row_id = "mro_" + stable_hash(
        {
            "source": str(source_path),
            "index": source_index,
            "candidate": repair_candidate_id,
            "route": route_id,
            "status": status,
        }
    )[:16]
    blocker_set = set(blockers)
    return {
        "schema": "minus_repair_outcome_training_row.v1",
        "row_id": row_id,
        "source_kind": source_kind,
        "source_path": str(source_path),
        "source_index": int(source_index),
        "track_id": track_id,
        "source_candidate_id": source_candidate_id,
        "repair_candidate_id": repair_candidate_id,
        "candidate_audio_path": candidate_audio_path,
        "stem_type": "minus",
        "route_id": route_id,
        "route_family": route_family,
        "status": str(status or ""),
        "label": label,
        "accepted_for_repair_gate": label == 1,
        "blocker_labels": list(blockers),
        "has_detail_damage_blocker": bool(blocker_set & DETAIL_DAMAGE_BLOCKERS),
        "has_leakage_blocker": bool(blocker_set & LEAKAGE_BLOCKERS),
        "feature_values": metrics,
        "parameters": dict(parameters or {}),
        "safe_for_repair_outcome_training": not reasons,
        "unsafe_reason": ";".join(reasons),
        "release_ready": False,
        "delivery_allowed": False,
        "can_publish_as_result": False,
        **(extra or {}),
    }


def rows_from_ralph_manifest(path: Path) -> list[dict[str, Any]]:
    manifest = read_json(path)
    filters = manifest.get("focus_window_filters") if isinstance(manifest.get("focus_window_filters"), dict) else {}
    track_id = str(filters.get("track_id") or "")
    source_candidate_id = str(filters.get("candidate_id") or "")
    rows: list[dict[str, Any]] = []
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        return rows
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        evaluation = candidate.get("evaluation") if isinstance(candidate.get("evaluation"), dict) else {}
        summary = evaluation.get("summary") if isinstance(evaluation.get("summary"), dict) else {}
        focus = evaluation.get("focus_evaluation") if isinstance(evaluation.get("focus_evaluation"), dict) else {}
        metrics = metric_bundle(
            score=evaluation.get("score"),
            avg_improve=evaluation.get("average_vocal_improvement_db"),
            peak_improve=evaluation.get("peak_corr_improvement"),
            worst_rms=summary.get("worst_rms_dip_db"),
            max_rms=summary.get("max_rms_boost_db"),
            worst_presence=summary.get("worst_presence_dip_db"),
            max_presence=summary.get("max_presence_boost_db"),
            worst_air=summary.get("worst_air_dip_db"),
            max_air=summary.get("max_air_boost_db"),
            focus_improve=focus.get("average_vocal_improvement_db"),
            focus_peak=focus.get("peak_corr_improvement"),
            focus_rms=focus.get("worst_rms_dip_db"),
        )
        repair_candidate_id = str(candidate.get("candidate_name") or evaluation.get("candidate_name") or f"candidate_{index:03d}")
        rows.append(
            build_common_row(
                source_kind="ralph_manifest_candidate",
                source_path=path,
                source_index=index,
                track_id=track_id,
                source_candidate_id=source_candidate_id,
                repair_candidate_id=repair_candidate_id,
                candidate_audio_path=str(candidate.get("artifact_path") or ""),
                route_id=repair_candidate_id,
                route_family="ralph_candidate_recipe",
                status=str(evaluation.get("status") or ""),
                blockers=blocker_labels(evaluation.get("blockers")),
                metrics=metrics,
                parameters={},
                extra={
                    "focus_evaluation_used_for_delivery": bool(manifest.get("focus_evaluation_used_for_delivery")),
                    "focus_window_source": str(manifest.get("focus_window_source") or ""),
                    "focus_window_count": len(manifest.get("focus_windows") or []) if isinstance(manifest.get("focus_windows"), list) else 0,
                },
            )
        )
    return rows


def rows_from_matrix(path: Path) -> list[dict[str, Any]]:
    family = route_family_from_matrix_path(path)
    sid = source_id(path)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(path)):
        route_id = str(row.get("id") or Path(str(row.get("path") or row.get("artifact") or f"row_{index:03d}")).stem)
        parameters = {
            key: row[key]
            for key in ("strength", "alpha", "scale")
            if key in row
        }
        metrics = metric_bundle(
            score=row.get("score"),
            avg_improve=row.get("avg_improve"),
            peak_improve=row.get("peak_improve"),
            worst_rms=row.get("worst_rms"),
            max_rms=row.get("max_rms"),
            worst_presence=row.get("worst_presence"),
            max_presence=row.get("max_presence"),
            worst_air=row.get("worst_air"),
            max_air=row.get("max_air"),
            focus_improve=row.get("focus_improve"),
            focus_peak=row.get("focus_peak"),
            focus_rms=row.get("focus_rms"),
        )
        candidate_audio_path = str(row.get("artifact") or row.get("path") or "")
        rows.append(
            build_common_row(
                source_kind="matrix_row",
                source_path=path,
                source_index=index,
                track_id="",
                source_candidate_id=str(row.get("path") or ""),
                repair_candidate_id=f"{sid}_{route_id}"[:96],
                candidate_audio_path=candidate_audio_path,
                route_id=route_id,
                route_family=family,
                status=str(row.get("status") or ""),
                blockers=blocker_labels(row.get("blockers")),
                metrics=metrics,
                parameters=parameters,
            )
        )
    return rows


def listen_preview_decision_status(manifest: dict[str, Any], row: dict[str, Any]) -> str:
    delivery_allowed = manifest.get("delivery_allowed") is True
    delivery_status = str(manifest.get("delivery_status") or "").strip().lower()
    presentation_status = str(manifest.get("presentation_status") or "").strip().lower()
    minus_decision = str(row.get("minus_decision") or row.get("decision") or "").strip().lower()
    if (
        delivery_allowed
        and delivery_status == "accepted"
        and presentation_status == "accepted_for_listen_pack"
        and minus_decision == "accept_candidate"
    ):
        return "accepted"
    return delivery_status or "blocked"


def listen_preview_metrics(manifest: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    detail_gate = row.get("minus_musical_detail_preservation_gate")
    detail_metrics = detail_gate.get("metrics") if isinstance(detail_gate, dict) and isinstance(detail_gate.get("metrics"), dict) else {}
    score_components = row.get("minus_score_components") if isinstance(row.get("minus_score_components"), dict) else {}
    return merge_metric_bundles(
        {
            "score": row.get("minus_score") or manifest.get("minus_score"),
            "overall_no_reference_score": row.get("overall_no_reference_score"),
            "confidence": row.get("confidence"),
            "window_count": row.get("window_count"),
        },
        {
            f"detail_gate_{key}": value
            for key, value in detail_metrics.items()
        },
        {
            f"minus_score_component_{key}": value
            for key, value in score_components.items()
        },
    )


def rows_from_listen_preview_manifest(path: Path) -> list[dict[str, Any]]:
    manifest = read_json(path)
    accepted_rows = manifest.get("accepted_listen_preview_rows")
    if not isinstance(accepted_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(accepted_rows):
        if not isinstance(row, dict):
            continue
        detail_gate = row.get("minus_musical_detail_preservation_gate")
        detail_blockers = (
            detail_gate.get("blockers")
            if isinstance(detail_gate, dict) and isinstance(detail_gate.get("blockers"), list)
            else []
        )
        blockers = stable_unique([*blocker_labels(row.get("minus_blockers")), *blocker_labels(detail_blockers)])
        candidate_id = str(row.get("candidate_id") or manifest.get("candidate_id") or f"listen_preview_{index:03d}")
        candidate_audio_path = str(
            row.get("minus_path")
            or row.get("instrumental_path")
            or manifest.get("selected_minus_path")
            or manifest.get("minus_path")
            or ""
        )
        rows.append(
            build_common_row(
                source_kind="listen_preview_manifest_row",
                source_path=path,
                source_index=index,
                track_id=str(row.get("track_id") or manifest.get("track_id") or ""),
                source_candidate_id=str(manifest.get("candidate_id") or candidate_id),
                repair_candidate_id=candidate_id,
                candidate_audio_path=candidate_audio_path,
                route_id=str(row.get("score_mode") or manifest.get("judge_mode") or "no_reference_minus_listen_preview"),
                route_family="accepted_listen_preview_repair_chain",
                status=listen_preview_decision_status(manifest, row),
                blockers=blockers,
                metrics=listen_preview_metrics(manifest, row),
                parameters={},
                extra={
                    "listen_preview_delivery_allowed": manifest.get("delivery_allowed") is True,
                    "listen_preview_delivery_status": str(manifest.get("delivery_status") or ""),
                    "listen_preview_presentation_status": str(manifest.get("presentation_status") or ""),
                    "listen_preview_release_status": str(manifest.get("release_status") or ""),
                    "listen_preview_manifest_type": str(manifest.get("manifest_type") or ""),
                    "result_judge_minus_decision": str(row.get("minus_decision") or row.get("decision") or ""),
                    "source_self_judge_manifest": str(
                        (manifest.get("source_self_judge_manifest") or {}).get("path")
                        if isinstance(manifest.get("source_self_judge_manifest"), dict)
                        else ""
                    ),
                    "source_result_judge_jsonl": str(
                        (manifest.get("source_result_judge_jsonl") or {}).get("path")
                        if isinstance(manifest.get("source_result_judge_jsonl"), dict)
                        else ""
                    ),
                    "source_selector_jsonl": str(
                        (manifest.get("source_selector_jsonl") or {}).get("path")
                        if isinstance(manifest.get("source_selector_jsonl"), dict)
                        else ""
                    ),
                },
            )
        )
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    safe_rows = [row for row in rows if row.get("safe_for_repair_outcome_training") is True]
    labels = Counter(str(row.get("label")) for row in safe_rows)
    blockers = Counter(blocker for row in rows for blocker in row.get("blocker_labels", []))
    families = Counter(str(row.get("route_family") or "") for row in rows)
    readiness_blockers: list[str] = []
    if len(safe_rows) < MIN_SAFE_REPAIR_ROWS:
        readiness_blockers.append("insufficient_safe_rows")
    if labels.get("1", 0) < MIN_POSITIVE_REPAIR_ROWS:
        readiness_blockers.append("insufficient_positive_repair_rows")
    if labels.get("0", 0) < MIN_NEGATIVE_REPAIR_ROWS:
        readiness_blockers.append("insufficient_negative_repair_rows")
    return {
        "rows_total": len(rows),
        "safe_rows": len(safe_rows),
        "unsafe_rows": len(rows) - len(safe_rows),
        "accepted_rows": int(labels.get("1", 0)),
        "rejected_rows": int(labels.get("0", 0)),
        "route_family_counts": dict(sorted(families.items())),
        "blocker_counts": dict(sorted(blockers.items())),
        "detail_damage_rows": sum(1 for row in rows if row.get("has_detail_damage_blocker") is True),
        "leakage_blocker_rows": sum(1 for row in rows if row.get("has_leakage_blocker") is True),
        "readiness_blockers": readiness_blockers,
        "can_train_repair_outcome_model": not readiness_blockers,
    }


def summary_text(manifest: dict[str, Any]) -> str:
    summary = manifest["summary"]
    lines = [
        "# Minus Repair Outcome Training View",
        "",
        f"- rows_total: `{summary['rows_total']}`",
        f"- safe_rows: `{summary['safe_rows']}`",
        f"- accepted_rows: `{summary['accepted_rows']}`",
        f"- rejected_rows: `{summary['rejected_rows']}`",
        f"- can_train_repair_outcome_model: `{str(summary['can_train_repair_outcome_model']).lower()}`",
        f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
        "",
        "## Top Blockers",
        "",
    ]
    blockers = summary.get("blocker_counts") or {}
    if blockers:
        lines.extend(f"- `{name}`: {count}" for name, count in sorted(blockers.items(), key=lambda item: (-item[1], item[0]))[:12])
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def build_minus_repair_outcome_training_view(
    *,
    output_dir: Path,
    ralph_manifests: Sequence[Path] = (),
    matrix_rows: Sequence[Path] = (),
    listen_preview_manifests: Sequence[Path] = (),
    command_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not ralph_manifests and not matrix_rows and not listen_preview_manifests:
        raise ValueError("missing_sources:provide_ralph_manifests_matrix_rows_or_listen_preview_manifests")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        ralph_manifest_count=len(ralph_manifests),
        matrix_rows_count=len(matrix_rows),
        listen_preview_manifest_count=len(listen_preview_manifests),
    )

    rows: list[dict[str, Any]] = []
    for path in ralph_manifests:
        loaded = rows_from_ralph_manifest(path)
        rows.extend(loaded)
        progress(progress_path, "loaded_ralph_manifest", path=str(path), rows_added=len(loaded))
    for path in matrix_rows:
        loaded = rows_from_matrix(path)
        rows.extend(loaded)
        progress(progress_path, "loaded_matrix_rows", path=str(path), rows_added=len(loaded))
    for path in listen_preview_manifests:
        loaded = rows_from_listen_preview_manifest(path)
        rows.extend(loaded)
        progress(progress_path, "loaded_listen_preview_manifest", path=str(path), rows_added=len(loaded))

    rows.sort(key=lambda row: (str(row.get("source_path") or ""), int(row.get("source_index") or 0), str(row.get("row_id") or "")))
    summary = summarize(rows)
    manifest = {
        "schema": SCHEMA,
        "mode": "minus_repair_outcome_training_view",
        "output_dir": str(output_dir),
        "ralph_manifests": [str(path) for path in ralph_manifests],
        "matrix_rows": [str(path) for path in matrix_rows],
        "listen_preview_manifests": [str(path) for path in listen_preview_manifests],
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "summary": summary,
        "command_args": list(command_args or []),
        **SAFETY_FLAGS,
    }
    write_jsonl(output_dir / ROWS_NAME, rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest), encoding="utf-8")
    progress(progress_path, "finish", rows_total=summary["rows_total"], safe_rows=summary["safe_rows"])
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build metadata-only minus repair outcome training rows.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ralph-manifests", type=Path, nargs="*", default=[])
    parser.add_argument("--matrix-rows", type=Path, nargs="*", default=[])
    parser.add_argument("--listen-preview-manifests", type=Path, nargs="*", default=[])
    parser.add_argument("--short-output-names", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_minus_repair_outcome_training_view.py",
        entrypoint_path="/app/tools/build_minus_repair_outcome_training_view.py",
    )
    args = parse_args(argv)
    build_minus_repair_outcome_training_view(
        output_dir=args.output_dir,
        ralph_manifests=args.ralph_manifests,
        matrix_rows=args.matrix_rows,
        listen_preview_manifests=args.listen_preview_manifests,
        command_args=list(sys.argv[1:] if argv is None else argv),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
