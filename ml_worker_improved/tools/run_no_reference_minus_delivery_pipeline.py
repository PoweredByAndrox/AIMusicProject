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

from tools.build_no_reference_minus_listen_preview_manifest import (  # noqa: E402
    build_manifest as build_minus_listen_preview_manifest,
    build_summary as build_minus_listen_preview_summary,
    write_json as write_contract_json,
)
from tools.apply_judge_guided_minus_vocal_leakage_repair import (  # noqa: E402
    build_judge_guided_minus_vocal_leakage_repair,
)
from tools.instrumental_self_judge_gate import MANIFEST_NAME as SELF_JUDGE_MANIFEST_NAME  # noqa: E402
from tools.instrumental_self_judge_gate import judge_instrumental  # noqa: E402
from tools.render_no_reference_minus_segment_switch_candidate import (  # noqa: E402
    BASE_FORBIDDEN_ACTIONS,
    build_minus_segment_switch_candidate,
    build_summary as build_render_summary,
    read_json as read_self_judge_json,
    write_json,
)
from tools.run_no_reference_candidate_selector import (  # noqa: E402
    select_candidates,
    write_outputs as write_selector_outputs,
)
from tools.run_no_reference_result_judge import (  # noqa: E402
    judge_candidate_rows,
    load_candidate_rows,
    write_outputs as write_result_judge_outputs,
)


PIPELINE_MANIFEST_NAME = "no_reference_minus_delivery_pipeline_manifest.json"
PIPELINE_SUMMARY_NAME = "no_reference_minus_delivery_pipeline_summary.md"

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "diagnostic_audio_exported": False,
    "audio_copied": False,
    "audio_exported_as_result": False,
    "release_ready": False,
}

ACCEPTABLE_CANDIDATE_STATUSES = {
    "rendered_minus_segment_switch_candidate",
    "existing_minus_candidate_proxy_pass",
}
SELF_JUDGE_LEAKAGE_ISSUES = {"vocal_projection_proxy", "vocal_stem_corr_proxy"}

START_TIME = time.monotonic()


def log_event(phase: str, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "phase": phase,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_inference_skipped": True,
        "gpu_inference_skipped_reason": "no_reference_minus_delivery_uses_existing_candidates_and_cpu_dsp_scoring",
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
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def discover_self_judge_manifests(roots: Iterable[Path]) -> list[Path]:
    discovered: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == SELF_JUDGE_MANIFEST_NAME:
            discovered.append(root)
            continue
        if root.is_dir():
            discovered.extend(sorted(root.rglob(SELF_JUDGE_MANIFEST_NAME)))
    return sorted({path.resolve(): path for path in discovered}.values(), key=lambda path: str(path))


def candidate_id_from_self_judge(payload: dict[str, Any]) -> str:
    candidate = payload.get("candidate")
    if isinstance(candidate, dict):
        label = candidate.get("label") or candidate.get("candidate_id")
        if label:
            return str(label)
    return str(payload.get("candidate_id") or payload.get("candidate_label") or "")


def blocker_count(payload: dict[str, Any]) -> int:
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    try:
        return int(overall.get("blocker_count") or 0)
    except (TypeError, ValueError):
        return 0


def blocker_ratio(payload: dict[str, Any]) -> float:
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    try:
        return float(overall.get("blocker_ratio") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def max_segment_severity(payload: dict[str, Any]) -> float:
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return 0.0
    severities: list[float] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        try:
            severities.append(float(segment.get("severity_score") or 0.0))
        except (TypeError, ValueError):
            continue
    return max(severities, default=0.0)


def has_self_judge_vocal_leakage_blocker(payload: dict[str, Any]) -> bool:
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return False
    for segment in segments:
        if not isinstance(segment, dict) or segment.get("blocker") is not True:
            continue
        if set(as_list(segment.get("issues"))) & SELF_JUDGE_LEAKAGE_ISSUES:
            return True
    return False


def choose_base_candidate_id(self_judge_payloads: list[dict[str, Any]]) -> str:
    candidates = [
        (
            blocker_count(payload),
            blocker_ratio(payload),
            max_segment_severity(payload),
            candidate_id_from_self_judge(payload),
        )
        for payload in self_judge_payloads
        if candidate_id_from_self_judge(payload)
    ]
    if not candidates:
        raise ValueError("no_self_judge_candidate_ids_found")
    repairable_candidates = [candidate for candidate in candidates if candidate[0] > 0]
    return sorted(repairable_candidates or candidates)[0][3]


def _matrix_row_identifiers(row: dict[str, Any]) -> set[str]:
    identifiers = {
        str(row.get("canonical_candidate_id") or ""),
        str(row.get("candidate_id") or ""),
    }
    aliases = row.get("aliases")
    if isinstance(aliases, list):
        identifiers.update(str(alias) for alias in aliases if str(alias))
    result_judge = row.get("result_judge")
    if isinstance(result_judge, dict):
        identifiers.add(str(result_judge.get("candidate_id") or ""))
    self_judge = row.get("self_judge")
    if isinstance(self_judge, dict):
        identifiers.add(str(self_judge.get("candidate_label") or ""))
    return {identifier for identifier in identifiers if identifier}


def choose_base_candidate_id_from_score_matrix(
    *,
    matrix_json: Path,
    self_judge_payloads: list[dict[str, Any]],
) -> str | None:
    payload = read_self_judge_json(matrix_json)
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    if not rows:
        return None
    labels = {candidate_id_from_self_judge(self_judge_payload): self_judge_payload for self_judge_payload in self_judge_payloads}
    ranked_rows = sorted(
        [row for row in rows if isinstance(row, dict)],
        key=lambda row: (
            int(row.get("rank") or 999999),
            -float(row.get("selection_priority") or 0.0),
            str(row.get("canonical_candidate_id") or ""),
        ),
    )
    actionable_next_steps = {
        "run_no_reference_result_judge_minus",
        "run_segment_router_for_self_blockers",
        "candidate_ready_for_listen_preview_contract",
        "run_self_judge_and_selector_before_delivery_contract",
        "accepted_for_listen_pack_not_release",
    }
    for row in ranked_rows:
        if str(row.get("next_action") or "") not in actionable_next_steps:
            continue
        canonical_candidate_id = str(row.get("canonical_candidate_id") or "")
        if canonical_candidate_id in labels:
            return canonical_candidate_id
        self_judge = row.get("self_judge")
        if isinstance(self_judge, dict):
            matrix_self_label = str(self_judge.get("candidate_label") or "")
            if matrix_self_label in labels:
                return matrix_self_label
        row_identifiers = _matrix_row_identifiers(row)
        matching_labels = [label for label in labels if label and label in row_identifiers]
        if matching_labels:
            return sorted(matching_labels, key=lambda label: (-len(label), label))[0]
    return None


def self_judge_payload_for_candidate(self_judge_payloads: list[dict[str, Any]], candidate_id: str) -> dict[str, Any] | None:
    for payload in self_judge_payloads:
        if candidate_id_from_self_judge(payload) == candidate_id:
            return payload
    return None


def build_existing_minus_candidate_manifest(
    *,
    self_judge_payload: dict[str, Any],
    output_dir: Path,
    candidate_id: str,
) -> dict[str, Any]:
    candidate = self_judge_payload.get("candidate") if isinstance(self_judge_payload.get("candidate"), dict) else {}
    source_audio_path = str(self_judge_payload.get("source_audio_path") or "")
    vocal_path = str(candidate.get("vocal_path") or "")
    minus_path = str(candidate.get("instrumental_path") or candidate.get("minus_path") or "")
    track_id = str(self_judge_payload.get("track_id") or Path(source_audio_path).stem or candidate_id)
    candidate_row = {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": source_audio_path,
        "source_audio_path": source_audio_path,
        "vocal_path": vocal_path,
        "instrumental_path": minus_path,
        "minus_path": minus_path,
        "selected_vocal_path": vocal_path,
        "selected_minus_path": minus_path,
        "candidate_family": "existing_minus_proxy_pass",
        "candidate_origin": "existing_instrumental_self_judge_proxy_pass",
        "base_candidate_id": candidate_id,
        "final_decision": "existing_minus_candidate",
        "requires_post_render_minus_self_judge": True,
        "requires_dense_post_reconciliation_judge": False,
        "needs_pair_reconciliation": False,
        "segment_blend_needed": False,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "delivery_allowed": False,
        "delivery_status": "blocked",
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "forbidden_actions": list(BASE_FORBIDDEN_ACTIONS),
        "is_complete_pair": bool(source_audio_path and vocal_path and minus_path),
    }
    candidate_row_jsonl = output_dir / "existing_minus_candidate_row.jsonl"
    write_jsonl(candidate_row_jsonl, [candidate_row])
    return {
        "manifest_type": "no_reference_minus_existing_candidate",
        "track_id": track_id,
        "base_candidate_id": candidate_id,
        "render_status": "existing_minus_candidate_proxy_pass",
        "dominant_failure_reason": "",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "preview_only_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_post_render_minus_self_judge": True,
        "requires_dense_post_reconciliation_judge": False,
        "selected_segment_count": 0,
        "overlay_segment_count": 0,
        "blocked_segment_count": 0,
        "selected_segments": [],
        "blocked_segments": [],
        "rendered_minus_candidate_path": minus_path,
        "candidate_row_jsonl": str(candidate_row_jsonl),
        "candidate_row": candidate_row,
        "source_self_judge_manifest_paths": [str(self_judge_payload.get("source_self_judge_manifest_path") or "")],
        "safety_flags": dict(SAFETY_FLAGS),
        "forbidden_actions": list(BASE_FORBIDDEN_ACTIONS),
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
    self_judge_manifest_paths: list[Path],
    base_candidate_id: str,
    reason: str,
    render_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "manifest_type": "no_reference_minus_delivery_pipeline",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": reason,
        "stem_type": "minus",
        "base_candidate_id": base_candidate_id,
        "source_self_judge_manifest_paths": source_paths(self_judge_manifest_paths),
        "render_manifest_path": str(output_dir / "minus_segment_switch_manifest.json") if render_manifest else "",
        "render_status": str((render_manifest or {}).get("render_status") or ""),
        "final_listen_preview_manifest": "",
        "accepted_count": 0,
        "skipped_count": 1,
        "forbidden_actions": list(BASE_FORBIDDEN_ACTIONS),
        "safety": dict(SAFETY_FLAGS),
    }


def missing_audio_inputs(
    *,
    source_audio: Path,
    candidate_minus: Path,
    candidate_vocal: Path | None,
) -> list[str]:
    paths = [source_audio, candidate_minus]
    if candidate_vocal is not None:
        paths.append(candidate_vocal)
    return [str(path) for path in paths if not str(path) or not path.exists()]


def accepted_pipeline_manifest(
    *,
    output_dir: Path,
    self_judge_manifest_paths: list[Path],
    base_candidate_id: str,
    render_manifest: dict[str, Any],
    listen_manifest: dict[str, Any],
    candidate_manifest_json: Path | None = None,
    post_candidate_self_judge_manifest: Path | None = None,
    result_judge_jsonl: Path | None = None,
    selector_jsonl: Path | None = None,
    repair_manifest_json: Path | None = None,
) -> dict[str, Any]:
    default_candidate_manifest = output_dir / "minus_segment_switch_manifest.json"
    default_self_judge = output_dir / "post_render_self_judge" / SELF_JUDGE_MANIFEST_NAME
    default_result_judge = output_dir / "no_reference_result_judge" / "no_reference_result_judge.jsonl"
    default_selector = output_dir / "no_reference_candidate_selector" / "no_reference_candidate_selection.jsonl"
    return {
        "manifest_type": "no_reference_minus_delivery_pipeline",
        "delivery_status": listen_manifest.get("delivery_status"),
        "delivery_allowed": listen_manifest.get("delivery_allowed") is True,
        "presentation_status": listen_manifest.get("presentation_status"),
        "release_status": listen_manifest.get("release_status"),
        "dominant_failure_reason": listen_manifest.get("dominant_failure_reason") or "",
        "stem_type": "minus",
        "base_candidate_id": base_candidate_id,
        "source_self_judge_manifest_paths": source_paths(self_judge_manifest_paths),
        "render_manifest_path": str(output_dir / "minus_segment_switch_manifest.json"),
        "candidate_manifest_path": str(candidate_manifest_json or default_candidate_manifest),
        "repair_manifest_path": str(repair_manifest_json or ""),
        "render_status": render_manifest.get("render_status"),
        "post_render_self_judge_manifest": str(default_self_judge),
        "post_candidate_self_judge_manifest": str(post_candidate_self_judge_manifest or default_self_judge),
        "result_judge_jsonl": str(result_judge_jsonl or default_result_judge),
        "selector_jsonl": str(selector_jsonl or default_selector),
        "final_listen_preview_manifest": str(output_dir / "no_reference_minus_listen_preview_manifest.json"),
        "accepted_count": int(listen_manifest.get("accepted_count") or 0),
        "skipped_count": int(listen_manifest.get("skipped_count") or 0),
        "selected_minus_path": listen_manifest.get("selected_minus_path") or "",
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "forbidden_actions": list(BASE_FORBIDDEN_ACTIONS),
        "safety": dict(SAFETY_FLAGS),
    }


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Minus Delivery Pipeline Summary",
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


def run_minus_delivery_pipeline(
    *,
    self_judge_manifest_paths: list[Path],
    output_dir: Path,
    base_candidate_id: str | None,
    candidate_score_matrix_json: Path | None = None,
    source_audio_path: Path | None = None,
    crossfade_ms: int = 80,
    segment_sec: float = 6.0,
    allow_repairable_blocked_alternate: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_event(
        "minus_delivery_pipeline",
        "start",
        output_dir=str(output_dir),
        self_judge_manifest_count=len(self_judge_manifest_paths),
        expected_acceleration_mode="cpu_dsp_scoring",
        allow_repairable_blocked_alternate=allow_repairable_blocked_alternate,
    )
    self_judge_payloads = [read_self_judge_json(path) for path in self_judge_manifest_paths]
    base_selection_source = "explicit_base_candidate_id" if base_candidate_id else "self_judge_heuristic"
    matrix_base_candidate_id = None
    if base_candidate_id is None and candidate_score_matrix_json is not None:
        matrix_base_candidate_id = choose_base_candidate_id_from_score_matrix(
            matrix_json=candidate_score_matrix_json,
            self_judge_payloads=self_judge_payloads,
        )
        if matrix_base_candidate_id:
            base_selection_source = "candidate_score_matrix"
    selected_base_candidate_id = base_candidate_id or matrix_base_candidate_id or choose_base_candidate_id(self_judge_payloads)
    log_event(
        "base_selection",
        "end",
        base_candidate_id=selected_base_candidate_id,
        base_selection_source=base_selection_source,
        candidate_score_matrix_json=str(candidate_score_matrix_json or ""),
    )
    base_self_judge_payload = self_judge_payload_for_candidate(self_judge_payloads, selected_base_candidate_id)
    if base_self_judge_payload is not None and blocker_count(base_self_judge_payload) == 0:
        render_manifest = build_existing_minus_candidate_manifest(
            self_judge_payload=base_self_judge_payload,
            output_dir=output_dir,
            candidate_id=selected_base_candidate_id,
        )
    else:
        render_manifest = build_minus_segment_switch_candidate(
            self_judge_payloads,
            output_dir=output_dir,
            base_candidate_id=selected_base_candidate_id,
            crossfade_ms=crossfade_ms,
            allow_repairable_blocked_alternate=allow_repairable_blocked_alternate,
        )
    render_manifest_json = output_dir / "minus_segment_switch_manifest.json"
    render_summary_md = output_dir / "minus_segment_switch_summary.md"
    write_json(render_manifest_json, render_manifest)
    render_summary_md.write_text(build_render_summary(render_manifest), encoding="utf-8")
    log_event(
        "minus_candidate_prepare",
        "end",
        base_candidate_id=selected_base_candidate_id,
        render_status=str(render_manifest.get("render_status") or ""),
        output_manifest=str(render_manifest_json),
    )

    if render_manifest.get("render_status") not in ACCEPTABLE_CANDIDATE_STATUSES:
        manifest = blocked_pipeline_manifest(
            output_dir=output_dir,
            self_judge_manifest_paths=self_judge_manifest_paths,
            base_candidate_id=selected_base_candidate_id,
            reason=str(render_manifest.get("dominant_failure_reason") or render_manifest.get("render_status") or "render_blocked"),
            render_manifest=render_manifest,
        )
        annotate_base_selection(
            manifest,
            base_selection_source=base_selection_source,
            candidate_score_matrix_json=candidate_score_matrix_json,
        )
        write_pipeline_outputs(output_dir, manifest)
        return manifest

    candidate_row = render_manifest.get("candidate_row") if isinstance(render_manifest.get("candidate_row"), dict) else {}
    candidate_row_jsonl = Path(str(render_manifest.get("candidate_row_jsonl") or output_dir / "minus_segment_switch_candidate_row.jsonl"))
    source_audio = source_audio_path or Path(str(candidate_row.get("source_audio_path") or candidate_row.get("mix_path") or ""))
    candidate_minus = Path(str(candidate_row.get("selected_minus_path") or candidate_row.get("minus_path") or ""))
    candidate_vocal_value = str(candidate_row.get("selected_vocal_path") or candidate_row.get("vocal_path") or "")
    candidate_vocal = Path(candidate_vocal_value) if candidate_vocal_value else None
    candidate_id = str(candidate_row.get("candidate_id") or "minus_segment_switch_candidate")
    missing_inputs = missing_audio_inputs(
        source_audio=source_audio,
        candidate_minus=candidate_minus,
        candidate_vocal=candidate_vocal,
    )
    if missing_inputs:
        manifest = blocked_pipeline_manifest(
            output_dir=output_dir,
            self_judge_manifest_paths=self_judge_manifest_paths,
            base_candidate_id=selected_base_candidate_id,
            reason="missing_audio_path_for_internal_analysis",
            render_manifest=render_manifest,
        )
        manifest["missing_audio_inputs"] = missing_inputs
        annotate_base_selection(
            manifest,
            base_selection_source=base_selection_source,
            candidate_score_matrix_json=candidate_score_matrix_json,
        )
        write_pipeline_outputs(output_dir, manifest)
        return manifest

    self_judge_dir = output_dir / "post_render_self_judge"
    log_event(
        "post_render_self_judge",
        "start",
        candidate_id=candidate_id,
        segment_sec=segment_sec,
        output_manifest=str(self_judge_dir / SELF_JUDGE_MANIFEST_NAME),
    )
    judge_instrumental(
        source_audio_path=source_audio,
        candidate_instrumental_path=candidate_minus,
        output_dir=self_judge_dir,
        candidate_vocal_path=candidate_vocal,
        candidate_label=candidate_id,
        segment_sec=segment_sec,
    )
    post_self_judge_manifest = self_judge_dir / SELF_JUDGE_MANIFEST_NAME
    log_event(
        "post_render_self_judge",
        "end",
        candidate_id=candidate_id,
        output_manifest=str(post_self_judge_manifest),
    )
    candidate_manifest_json = render_manifest_json
    post_candidate_self_judge_manifest = post_self_judge_manifest
    repair_manifest_json: Path | None = None
    post_self_judge_payload = read_self_judge_json(post_self_judge_manifest)
    if has_self_judge_vocal_leakage_blocker(post_self_judge_payload):
        repair_dir = output_dir / "post_render_leakage_repair"
        log_event(
            "post_render_leakage_repair",
            "start",
            candidate_id=candidate_id,
            judge_manifest=str(post_self_judge_manifest),
            output_dir=str(repair_dir),
        )
        try:
            repair_manifest = build_judge_guided_minus_vocal_leakage_repair(
                argparse.Namespace(
                    raw_audio_path=source_audio,
                    base_minus_path=candidate_minus,
                    primary_vocals_path=candidate_vocal,
                    windowed_judge_jsonl=post_self_judge_manifest,
                    output_dir=repair_dir,
                    candidate_id=candidate_id,
                    track_id=str(candidate_row.get("track_id") or Path(source_audio).stem or candidate_id),
                    listen_dir=None,
                    listen_label="minus_judge_guided_leakage_repair.flac",
                    leak_threshold=0.60,
                    strength=0.85,
                    max_projection_gain=0.75,
                    gate_smoothing_sec=0.03,
                    window_margin_sec=0.04,
                    band_low_hz=160.0,
                    band_high_hz=8000.0,
                    projection_mode="segment_local",
                )
            )
        except Exception as exc:
            manifest = blocked_pipeline_manifest(
                output_dir=output_dir,
                self_judge_manifest_paths=self_judge_manifest_paths,
                base_candidate_id=selected_base_candidate_id,
                reason="post_render_leakage_repair_failed",
                render_manifest=render_manifest,
            )
            manifest["repair_error"] = str(exc)
            annotate_base_selection(
                manifest,
                base_selection_source=base_selection_source,
                candidate_score_matrix_json=candidate_score_matrix_json,
            )
            write_pipeline_outputs(output_dir, manifest)
            return manifest
        repair_manifest_json = repair_dir / "judge_guided_minus_vocal_leakage_repair_manifest.json"
        repaired_row = repair_manifest.get("candidate_row") if isinstance(repair_manifest.get("candidate_row"), dict) else {}
        candidate_row = repaired_row
        candidate_row_jsonl = Path(str(repair_manifest.get("candidate_row_jsonl") or repair_dir / "candidate_pair.jsonl"))
        candidate_minus = Path(str(candidate_row.get("selected_minus_path") or candidate_row.get("minus_path") or ""))
        candidate_vocal_value = str(candidate_row.get("selected_vocal_path") or candidate_row.get("vocal_path") or "")
        candidate_vocal = Path(candidate_vocal_value) if candidate_vocal_value else None
        candidate_id = str(candidate_row.get("candidate_id") or repair_manifest.get("candidate_id") or candidate_id)
        candidate_manifest_json = repair_manifest_json
        missing_repair_inputs = missing_audio_inputs(
            source_audio=source_audio,
            candidate_minus=candidate_minus,
            candidate_vocal=candidate_vocal,
        )
        if missing_repair_inputs:
            manifest = blocked_pipeline_manifest(
                output_dir=output_dir,
                self_judge_manifest_paths=self_judge_manifest_paths,
                base_candidate_id=selected_base_candidate_id,
                reason="missing_audio_path_after_leakage_repair",
                render_manifest=render_manifest,
            )
            manifest["repair_manifest_path"] = str(repair_manifest_json)
            manifest["missing_audio_inputs"] = missing_repair_inputs
            annotate_base_selection(
                manifest,
                base_selection_source=base_selection_source,
                candidate_score_matrix_json=candidate_score_matrix_json,
            )
            write_pipeline_outputs(output_dir, manifest)
            return manifest
        log_event(
            "post_render_leakage_repair",
            "end",
            candidate_id=candidate_id,
            output_manifest=str(repair_manifest_json),
            selected_window_count=int(repair_manifest.get("repair", {}).get("selected_window_count") or 0)
            if isinstance(repair_manifest.get("repair"), dict)
            else 0,
        )

        repair_self_judge_dir = output_dir / "post_repair_self_judge"
        log_event(
            "post_repair_self_judge",
            "start",
            candidate_id=candidate_id,
            segment_sec=segment_sec,
            output_manifest=str(repair_self_judge_dir / SELF_JUDGE_MANIFEST_NAME),
        )
        judge_instrumental(
            source_audio_path=source_audio,
            candidate_instrumental_path=candidate_minus,
            output_dir=repair_self_judge_dir,
            candidate_vocal_path=candidate_vocal,
            candidate_label=candidate_id,
            segment_sec=segment_sec,
        )
        post_candidate_self_judge_manifest = repair_self_judge_dir / SELF_JUDGE_MANIFEST_NAME
        log_event(
            "post_repair_self_judge",
            "end",
            candidate_id=candidate_id,
            output_manifest=str(post_candidate_self_judge_manifest),
        )

    result_judge_dir = output_dir / "no_reference_result_judge"
    log_event(
        "result_judge",
        "start",
        score_mode="minus",
        candidate_row_jsonl=str(candidate_row_jsonl),
        output_dir=str(result_judge_dir),
    )
    result = judge_candidate_rows(load_candidate_rows(candidate_row_jsonl, []), score_mode="minus")
    write_result_judge_outputs(result, result_judge_dir)
    result_judge_jsonl = result_judge_dir / "no_reference_result_judge.jsonl"
    log_event(
        "result_judge",
        "end",
        score_mode="minus",
        output_jsonl=str(result_judge_jsonl),
        candidates_judged=int(result.get("summary", {}).get("candidates_judged") or 0),
    )

    selector_dir = output_dir / "no_reference_candidate_selector"
    log_event("selector", "start", score_mode="minus", output_dir=str(selector_dir))
    selector_result = select_candidates(list(iter_jsonl(result_judge_jsonl)), score_mode="minus")
    write_selector_outputs(selector_result, selector_dir)
    selector_jsonl = selector_dir / "no_reference_candidate_selection.jsonl"
    log_event(
        "selector",
        "end",
        score_mode="minus",
        output_jsonl=str(selector_jsonl),
        selected_tracks=int(selector_result.get("summary", {}).get("total_tracks") or 0),
    )

    listen_manifest_json = output_dir / "no_reference_minus_listen_preview_manifest.json"
    listen_summary_md = output_dir / "no_reference_minus_listen_preview_summary.md"
    listen_manifest = build_minus_listen_preview_manifest(
        candidate_manifest_json=candidate_manifest_json,
        self_judge_manifest_json=post_candidate_self_judge_manifest,
        result_judge_jsonl=result_judge_jsonl,
        selector_jsonl=selector_jsonl,
    )
    write_contract_json(listen_manifest_json, listen_manifest)
    listen_summary_md.write_text(build_minus_listen_preview_summary(listen_manifest), encoding="utf-8")
    log_event(
        "minus_listen_preview_manifest",
        "end",
        delivery_allowed=listen_manifest.get("delivery_allowed") is True,
        delivery_status=str(listen_manifest.get("delivery_status") or ""),
        output_manifest=str(listen_manifest_json),
    )

    if listen_manifest.get("delivery_allowed") is not True:
        manifest = blocked_pipeline_manifest(
            output_dir=output_dir,
            self_judge_manifest_paths=self_judge_manifest_paths,
            base_candidate_id=selected_base_candidate_id,
            reason=str(listen_manifest.get("dominant_failure_reason") or "minus_listen_preview_not_accepted"),
            render_manifest=render_manifest,
        )
        manifest["final_listen_preview_manifest"] = str(listen_manifest_json)
        manifest["skipped_count"] = int(listen_manifest.get("skipped_count") or 1)
        manifest["candidate_manifest_path"] = str(candidate_manifest_json)
        manifest["repair_manifest_path"] = str(repair_manifest_json or "")
        manifest["post_candidate_self_judge_manifest"] = str(post_candidate_self_judge_manifest)
        manifest["result_judge_jsonl"] = str(result_judge_jsonl)
        manifest["selector_jsonl"] = str(selector_jsonl)
        annotate_base_selection(
            manifest,
            base_selection_source=base_selection_source,
            candidate_score_matrix_json=candidate_score_matrix_json,
        )
        write_pipeline_outputs(output_dir, manifest)
        return manifest

    manifest = accepted_pipeline_manifest(
        output_dir=output_dir,
        self_judge_manifest_paths=self_judge_manifest_paths,
        base_candidate_id=selected_base_candidate_id,
        render_manifest=render_manifest,
        listen_manifest=listen_manifest,
        candidate_manifest_json=candidate_manifest_json,
        post_candidate_self_judge_manifest=post_candidate_self_judge_manifest,
        result_judge_jsonl=result_judge_jsonl,
        selector_jsonl=selector_jsonl,
        repair_manifest_json=repair_manifest_json,
    )
    annotate_base_selection(
        manifest,
        base_selection_source=base_selection_source,
        candidate_score_matrix_json=candidate_score_matrix_json,
    )
    write_pipeline_outputs(output_dir, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the no-reference minus-only delivery pipeline from existing self-judge manifests."
    )
    parser.add_argument("--self-judge-manifest", type=Path, action="append", default=[])
    parser.add_argument("--self-judge-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-candidate-id")
    parser.add_argument("--candidate-score-matrix-json", type=Path)
    parser.add_argument("--source-audio", type=Path)
    parser.add_argument("--crossfade-ms", type=int, default=80)
    parser.add_argument("--segment-sec", type=float, default=6.0)
    parser.add_argument("--allow-repairable-blocked-alternate", action="store_true")
    args = parser.parse_args(argv)

    self_judge_manifest_paths = sorted(
        {path.resolve(): path for path in [*args.self_judge_manifest, *discover_self_judge_manifests(args.self_judge_root)]}.values(),
        key=lambda path: str(path),
    )
    if not self_judge_manifest_paths:
        parser.error("at least one --self-judge-manifest or --self-judge-root with manifests is required")

    manifest = run_minus_delivery_pipeline(
        self_judge_manifest_paths=self_judge_manifest_paths,
        output_dir=args.output_dir,
        base_candidate_id=str(args.base_candidate_id) if args.base_candidate_id else None,
        candidate_score_matrix_json=args.candidate_score_matrix_json,
        source_audio_path=args.source_audio,
        crossfade_ms=int(args.crossfade_ms),
        segment_sec=float(args.segment_sec),
        allow_repairable_blocked_alternate=bool(args.allow_repairable_blocked_alternate),
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
