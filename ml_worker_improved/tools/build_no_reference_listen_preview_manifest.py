from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_status_contract import (
    ArtifactContractError,
    blocking_labels_from_values,
    can_promote_to_listen_result,
    sha256_file,
    stable_track_hash,
)


ACCEPTED_FINAL_DECISIONS = {
    "mode_specific_pair_selected",
    "optimized_candidate_selected",
    "original_candidate_kept",
    "pair_reconciliation_candidate",
    "phase_corrected_candidate_selected",
}

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
    "audio_exported": False,
    "release_ready": False,
}


MODES = ("vocal", "minus", "both")
MIN_DENSE_WINDOW_COUNT = 5


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"jsonl_row_must_be_object:{path}")
                yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_must_be_object:{path}")
    return payload


def stable_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def dense_gate_contract(mode: str) -> tuple[str, str, str, str]:
    if mode == "vocal":
        return ("vocal_score", "vocal_decision", "vocal_blockers", "accept_candidate")
    if mode == "minus":
        return ("minus_score", "minus_decision", "minus_blockers", "accept_candidate")
    return ("pair_compatibility_score", "pair_decision", "pair_blockers", "pair_compatible")


def dense_gate_min_window_count(gate_manifest: dict[str, Any]) -> int:
    min_window_count = as_int(gate_manifest.get("min_window_count") or MIN_DENSE_WINDOW_COUNT, default=MIN_DENSE_WINDOW_COUNT)
    return max(min_window_count, MIN_DENSE_WINDOW_COUNT)


def dense_gate_mode_failures(mode: str, gate: Any, *, min_window_count: int) -> list[str]:
    if not isinstance(gate, dict):
        return [f"missing_dense_delivery_gate_mode:{mode}"]

    score_key, decision_key, blocker_key, accepted_decision = dense_gate_contract(mode)
    failures: list[str] = []
    blocking_labels = blocking_labels_from_values(gate.get("reasons"), gate.get("decision"))
    failures.extend(f"{mode}_blocking_semantic_label:{label}" for label in blocking_labels)
    if str(gate.get("decision") or "") != "accept_candidate":
        failures.append(f"{mode}_decision_not_accepted")
    if as_int(gate.get("window_count")) < min_window_count:
        failures.append(f"{mode}_insufficient_window_count")
    if score_key not in gate or gate.get(score_key) in (None, ""):
        failures.append(f"missing_mode_specific_score:{score_key}")
    if decision_key not in gate or gate.get(decision_key) in (None, ""):
        failures.append(f"missing_mode_specific_decision:{decision_key}")
    elif str(gate.get(decision_key) or "") != accepted_decision:
        failures.append(f"{mode}_{decision_key}_not_accepted")
    if blocker_key not in gate or gate.get(blocker_key) is None:
        failures.append(f"missing_mode_specific_blockers:{blocker_key}")
    elif as_list(gate.get(blocker_key)):
        failures.append(f"{mode}_{blocker_key}_present")
    return failures


def dense_gate_evidence_failures(row: Any, *, min_window_count: int) -> list[str]:
    if not isinstance(row, dict):
        return ["accepted_row_must_be_object"]
    gate = row.get("dense_delivery_gate")
    if not isinstance(gate, dict):
        return ["missing_dense_delivery_gate"]

    failures: list[str] = []
    for mode in MODES:
        failures.extend(dense_gate_mode_failures(mode, gate.get(mode), min_window_count=min_window_count))
    return stable_unique(failures)


def dense_gate_rows_evidence_failures(rows: Sequence[Any], *, min_window_count: int) -> list[str]:
    failures: list[str] = []
    for index, row in enumerate(rows):
        for failure in dense_gate_evidence_failures(row, min_window_count=min_window_count):
            failures.append(f"row:{index}:{failure}")
    return stable_unique(failures)


def is_accepted_optimizer_row(row: dict[str, Any]) -> bool:
    if str(row.get("final_decision") or "") not in ACCEPTED_FINAL_DECISIONS:
        return False
    if str(row.get("final_decision") or "") == "pair_reconciliation_candidate" and row.get("requires_dense_post_reconciliation_judge") is not True:
        return False
    if str(row.get("final_decision") or "") == "phase_corrected_candidate_selected" and row.get("requires_post_correction_judge") is not True:
        return False
    if str(row.get("pair_compatibility_decision") or "") not in {"", "pair_compatible"}:
        return False
    if row.get("needs_pair_reconciliation") is True or row.get("segment_blend_needed") is True:
        return False
    return bool(str(row.get("selected_vocal_path") or row.get("vocal_path") or "").strip()) and bool(
        str(row.get("selected_minus_path") or row.get("instrumental_path") or row.get("minus_path") or "").strip()
    )


def listen_row_from_optimizer(row: dict[str, Any], source_manifest: Path) -> dict[str, Any]:
    vocal_path = str(row.get("selected_vocal_path") or row.get("vocal_path") or "")
    minus_path = str(row.get("selected_minus_path") or row.get("instrumental_path") or row.get("minus_path") or "")
    track_id = str(row.get("track_id") or "")
    track_hash = stable_track_hash(track_id, str(row.get("mix_path") or ""))
    return {
        "track_id": track_id,
        "track_hash": track_hash,
        "candidate_id": str(row.get("candidate_id") or row.get("selected_candidate_id") or row.get("input_candidate_id") or ""),
        "selected_vocal_candidate_id": str(row.get("selected_vocal_candidate_id") or row.get("selected_candidate_id") or ""),
        "selected_minus_candidate_id": str(row.get("selected_minus_candidate_id") or row.get("selected_candidate_id") or ""),
        "mix_path": str(row.get("mix_path") or ""),
        "vocal_path": vocal_path,
        "minus_path": minus_path,
        "instrumental_path": minus_path,
        "selected_vocal_score": float(row.get("selected_vocal_score") or 0.0),
        "selected_minus_score": float(row.get("selected_minus_score") or 0.0),
        "pair_compatibility_score": float(row.get("pair_compatibility_score") or 0.0),
        "pair_compatibility_decision": str(row.get("pair_compatibility_decision") or ""),
        "source_optimizer_result": str(source_manifest),
        "source_optimizer_result_sha256": sha256_file(source_manifest),
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
    }


def skipped_optimizer_rows(rows: list[dict[str, Any]], *, skip_reason: str) -> list[dict[str, Any]]:
    return [
        {
            "track_id": str(row.get("track_id") or ""),
            "candidate_id": str(row.get("candidate_id") or ""),
            "final_decision": str(row.get("final_decision") or ""),
            "skip_reason": skip_reason if is_accepted_optimizer_row(row) else "not_accepted_for_listen_preview",
        }
        for row in rows
    ]


def blocked_manifest(
    optimizer_result_jsonl: Path,
    rows: list[dict[str, Any]],
    *,
    reason: str,
    dense_delivery_gate_manifest_json: Path | None = None,
    dense_delivery_gate_evidence_failures: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "manifest_type": "no_reference_listen_preview",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": reason,
        "stem_type": "both",
        "judge_mode": "no_reference_dense_windowed",
        "source_manifest": str(optimizer_result_jsonl),
        "source_manifest_sha256": sha256_file(optimizer_result_jsonl),
        "dense_delivery_gate_manifest": str(dense_delivery_gate_manifest_json) if dense_delivery_gate_manifest_json else "",
        "dense_delivery_gate_manifest_sha256": sha256_file(dense_delivery_gate_manifest_json)
        if dense_delivery_gate_manifest_json and dense_delivery_gate_manifest_json.exists()
        else "",
        "accepted_count": 0,
        "skipped_count": len(rows),
        "accepted_listen_preview_rows": [],
        "skipped_rows": sorted(skipped_optimizer_rows(rows, skip_reason=reason), key=lambda item: (item["track_id"], item["candidate_id"])),
        "dense_delivery_gate_evidence_failures": list(dense_delivery_gate_evidence_failures),
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": dict(SAFETY_FLAGS),
    }


def dense_gate_is_source_for_optimizer(gate: dict[str, Any], optimizer_result_jsonl: Path) -> bool:
    expected_sha = sha256_file(optimizer_result_jsonl)
    gate_sha = str(gate.get("source_manifest_sha256") or "")
    return bool(gate_sha) and gate_sha == expected_sha


def build_manifest(optimizer_result_jsonl: Path, dense_delivery_gate_manifest_json: Path | None = None) -> dict[str, Any]:
    rows = list(iter_jsonl(optimizer_result_jsonl))
    if dense_delivery_gate_manifest_json is None:
        return blocked_manifest(optimizer_result_jsonl, rows, reason="missing_dense_delivery_gate")

    gate = read_json(dense_delivery_gate_manifest_json)
    if not dense_gate_is_source_for_optimizer(gate, optimizer_result_jsonl):
        return blocked_manifest(
            optimizer_result_jsonl,
            rows,
            reason="dense_delivery_gate_source_manifest_mismatch",
            dense_delivery_gate_manifest_json=dense_delivery_gate_manifest_json,
        )

    try:
        dense_promotable = can_promote_to_listen_result(dense_delivery_gate_manifest_json)
    except ArtifactContractError:
        dense_promotable = False
    accepted_rows = list(gate.get("accepted_listen_preview_rows") or [])
    skipped = list(gate.get("skipped_rows") or [])
    dense_evidence_failures = dense_gate_rows_evidence_failures(
        accepted_rows,
        min_window_count=dense_gate_min_window_count(gate),
    )
    has_dense_evidence = bool(accepted_rows) and not dense_evidence_failures
    delivery_allowed = bool(dense_promotable and accepted_rows and not skipped and has_dense_evidence)
    if not delivery_allowed:
        reason = "dense_delivery_gate_missing_mode_specific_evidence" if dense_evidence_failures else "dense_delivery_gate_not_promotable"
        return blocked_manifest(
            optimizer_result_jsonl,
            rows,
            reason=reason,
            dense_delivery_gate_manifest_json=dense_delivery_gate_manifest_json,
            dense_delivery_gate_evidence_failures=dense_evidence_failures,
        )

    return {
        "manifest_type": "no_reference_listen_preview",
        "delivery_status": "accepted",
        "delivery_allowed": delivery_allowed,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": "both",
        "judge_mode": "no_reference_dense_windowed",
        "source_manifest": str(optimizer_result_jsonl),
        "source_manifest_sha256": sha256_file(optimizer_result_jsonl),
        "dense_delivery_gate_manifest": str(dense_delivery_gate_manifest_json),
        "dense_delivery_gate_manifest_sha256": sha256_file(dense_delivery_gate_manifest_json),
        "accepted_count": len(accepted_rows),
        "skipped_count": 0,
        "accepted_listen_preview_rows": sorted(accepted_rows, key=lambda item: (item["track_id"], item["candidate_id"])),
        "skipped_rows": [],
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": dict(SAFETY_FLAGS),
    }


def build_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = list(manifest.get("accepted_listen_preview_rows") or [])
    final_decisions = Counter(str(row.get("pair_compatibility_decision") or "") for row in rows)
    return {
        "accepted_count": int(manifest.get("accepted_count") or 0),
        "skipped_count": int(manifest.get("skipped_count") or 0),
        "delivery_allowed": bool(manifest.get("delivery_allowed")),
        "presentation_status": str(manifest.get("presentation_status") or ""),
        "release_status": str(manifest.get("release_status") or ""),
        "can_mark_release_ready": bool(manifest.get("can_mark_release_ready")),
        "audio_copied": bool(manifest.get("audio_copied") or manifest.get("safety", {}).get("audio_copied")),
        "audio_exported": bool(manifest.get("audio_exported") or manifest.get("safety", {}).get("audio_exported")),
        "pair_compatibility_decisions": dict(sorted(final_decisions.items())),
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Listen Preview Manifest Summary",
        "",
        f"- accepted_count: {summary['accepted_count']}",
        f"- skipped_count: {summary['skipped_count']}",
        f"- delivery_allowed: {str(summary['delivery_allowed']).lower()}",
        f"- presentation_status: {summary['presentation_status']}",
        f"- release_status: {summary['release_status']}",
        f"- can_mark_release_ready: {str(summary['can_mark_release_ready']).lower()}",
        f"- audio_copied: {str(summary['audio_copied']).lower()}",
        f"- audio_exported: {str(summary['audio_exported']).lower()}",
        "",
        "## Pair Compatibility Decisions",
        "",
    ]
    for key, value in summary["pair_compatibility_decisions"].items():
        lines.append(f"- {key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a contract-safe no-reference listen-preview manifest.")
    parser.add_argument("--optimizer-result-jsonl", type=Path, required=True)
    parser.add_argument("--dense-delivery-gate-manifest-json", type=Path)
    parser.add_argument("--output-manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = build_manifest(args.optimizer_result_jsonl, dense_delivery_gate_manifest_json=args.dense_delivery_gate_manifest_json)
    write_json(args.output_manifest_json, manifest)
    try:
        promotable = can_promote_to_listen_result(args.output_manifest_json)
    except ArtifactContractError as exc:
        raise SystemExit(str(exc)) from exc
    if not promotable:
        raise SystemExit("generated_manifest_is_not_promotable")
    summary = build_summary(manifest)
    write_summary_md(args.summary_md, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
