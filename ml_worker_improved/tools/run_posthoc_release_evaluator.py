from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from tools import adaptive_minus_repair
from tools import adaptive_vocal_isolation


SUPPORTED_MANIFEST_NAMES = ("adaptive_vocal_manifest.json", "adaptive_manifest.json")
VOCAL_MODE = "adaptive_vocal_isolation_controller"
MINUS_MODE = "adaptive_minus_repair_controller"
MINUS_RELEASE_REQUIRED_METRICS = (
    "vocal_residue_corr",
    "vocal_to_instrumental_rms_ratio_db",
    "presence_band_ratio_drift_db",
    "air_band_ratio_drift_db",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _selected_candidate(manifest: dict[str, Any]) -> dict[str, Any]:
    selected_profile = manifest.get("selected_profile")
    for candidate in _as_list(manifest.get("candidates")):
        if isinstance(candidate, dict) and candidate.get("profile_name") == selected_profile:
            return candidate
    return {}


def _available_candidate_names(manifest: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for candidate in _as_list(manifest.get("candidates")):
        if isinstance(candidate, dict) and candidate.get("profile_name") is not None:
            names.add(str(candidate["profile_name"]))
    return names


def _skipped_vocal_candidates(manifest: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    skipped_by_name: dict[str, list[dict[str, str]]] = {}
    for candidate_name, manifest_key in (
        ("issue_aware_segment_selection", "skipped_issue_aware_segment_selection"),
        ("segment_issue_repair", "skipped_segment_issue_repair"),
        ("no_vocal_leakage_cleanup", "skipped_no_vocal_leakage_cleanup"),
        ("vocal_harshness_guard", "skipped_vocal_harshness_guard"),
    ):
        records: list[dict[str, str]] = []
        for record in _as_list(manifest.get(manifest_key)):
            if isinstance(record, dict):
                records.append({str(key): str(value) for key, value in record.items()})
        skipped_by_name[candidate_name] = records
    return skipped_by_name


def _blocker_reasons(blockers: list[Any]) -> list[str]:
    reasons: list[str] = []
    for blocker in blockers:
        if isinstance(blocker, dict) and blocker.get("reason") is not None:
            reasons.append(str(blocker["reason"]))
    return sorted(set(reasons))


def _blocker_labels(blockers: list[Any]) -> list[str]:
    labels: set[str] = set()
    for blocker in blockers:
        if not isinstance(blocker, dict):
            continue
        if blocker.get("label") is not None:
            labels.add(str(blocker["label"]))
        for label in _as_list(blocker.get("labels")):
            labels.add(str(label))
    return sorted(labels)


def _worst_windows(ledger: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    windows = [
        window
        for window in _as_list(ledger.get("windows"))
        if isinstance(window, dict)
    ]
    windows.sort(key=lambda window: float(window.get("severity_score", 0.0)), reverse=True)
    return [
        {
            "start_sec": window.get("start_sec"),
            "end_sec": window.get("end_sec"),
            "severity_score": window.get("severity_score"),
            "labels": _as_list(window.get("labels")),
            "release_decision": window.get("release_decision"),
        }
        for window in windows[:limit]
    ]


def _build_vocal_release_fields(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    rebuilt_fields: list[str] = []
    issue_report = manifest.get("selected_segment_issue_report")
    if not isinstance(issue_report, dict):
        issue_report = None

    action_contract = manifest.get("detected_issue_action_contract")
    if not isinstance(action_contract, dict):
        action_contract = adaptive_vocal_isolation.build_detected_issue_action_contract(
            issue_report=issue_report,
            available_candidate_names=_available_candidate_names(manifest),
            skipped_candidates=_skipped_vocal_candidates(manifest),
        )
        rebuilt_fields.append("detected_issue_action_contract")

    selected_metrics = _as_dict(_selected_candidate(manifest).get("metrics"))
    defect_gate = manifest.get("defect_effectiveness_gate")
    if not isinstance(defect_gate, dict):
        defect_gate = adaptive_vocal_isolation.build_defect_effectiveness_gate(
            issue_report=issue_report,
            action_contract=action_contract,
            selected_metrics=selected_metrics,
        )
        rebuilt_fields.append("defect_effectiveness_gate")

    issue_ledger = manifest.get("issue_effectiveness_ledger")
    if not isinstance(issue_ledger, dict):
        issue_ledger = adaptive_vocal_isolation.build_issue_effectiveness_ledger(
            issue_report=issue_report,
            action_contract=action_contract,
            defect_effectiveness_gate=defect_gate,
        )
        rebuilt_fields.append("issue_effectiveness_ledger")

    return action_contract, defect_gate, issue_ledger, rebuilt_fields


def _selected_artifacts(manifest: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(manifest.get("selected_artifacts"))


def evaluate_vocal_manifest(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    action_contract, defect_gate, issue_ledger, rebuilt_fields = _build_vocal_release_fields(manifest)
    selected_artifacts = _selected_artifacts(manifest)
    artifact_release_status = selected_artifacts.get("release_status")
    release_status = defect_gate.get("release_status", "diagnostic_only")
    blockers = _as_list(defect_gate.get("blockers"))
    contract_summary = _as_dict(action_contract.get("summary"))
    ledger_summary = _as_dict(issue_ledger.get("summary"))
    return {
        "kind": "vocal",
        "manifest_path": str(manifest_path),
        "source_audio_path": manifest.get("source_audio_path"),
        "selected_profile": manifest.get("selected_profile"),
        "selected_score": manifest.get("selected_score"),
        "release_status": release_status,
        "gate_status": defect_gate.get("status", "blocked_for_listening"),
        "contract_status": action_contract.get("status", "missing_contract"),
        "artifact_release_status": artifact_release_status,
        "release_status_drift": bool(artifact_release_status is not None and artifact_release_status != release_status),
        "rebuilt_fields": rebuilt_fields,
        "blocker_reasons": _blocker_reasons(blockers),
        "blocker_labels": _blocker_labels(blockers),
        "uncovered_labels": _as_list(contract_summary.get("uncovered_labels")),
        "covered_labels": _as_list(contract_summary.get("covered_labels")),
        "rejected_labels": _as_list(contract_summary.get("rejected_labels")),
        "issue_segment_count": contract_summary.get("issue_segment_count", 0),
        "blocked_window_count": ledger_summary.get("blocked_window_count", 0),
        "worst_windows": _worst_windows(issue_ledger),
        "selected_output_path": selected_artifacts.get("selected_vocals_flac"),
        "listen_output_path": selected_artifacts.get("listen_selected_vocals_flac"),
        "listen_issue_clips_manifest": selected_artifacts.get("listen_segment_issue_clips_manifest"),
    }


def _build_minus_release_fields(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    selected_gate = manifest.get("selected_minus_release_gate")
    selected_candidate = _selected_candidate(manifest)
    metrics = selected_candidate.get("metrics")
    missing_metric_names = list(MINUS_RELEASE_REQUIRED_METRICS)
    if isinstance(metrics, dict):
        missing_metric_names = [
            key
            for key in MINUS_RELEASE_REQUIRED_METRICS
            if key not in metrics
            or not isinstance(metrics.get(key), (int, float))
            or not math.isfinite(float(metrics.get(key)))
        ]
    missing_metrics = bool(missing_metric_names)
    if missing_metrics:
        return (
            {
                "schema_version": 1,
                "mode": "minus_release_gate",
                "status": "blocked_for_listening",
                "release_status": "diagnostic_only",
                "summary": {
                    "blocker_count": 1,
                    "focus_blocker_count": 0,
                },
                "blockers": [
                    {
                        "label": "missing_minus_release_metrics",
                        "reason": "missing_minus_release_metrics",
                        "required_metrics": list(MINUS_RELEASE_REQUIRED_METRICS),
                        "missing_metrics": missing_metric_names,
                    }
                ],
            },
            ["selected_minus_release_gate"],
        )
    if isinstance(selected_gate, dict):
        return selected_gate, []

    selected_gate = adaptive_minus_repair.build_minus_release_gate(
        metrics,
        _as_dict(selected_candidate.get("focus_metrics")),
    )
    return selected_gate, ["selected_minus_release_gate"]


def evaluate_minus_manifest(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    selected_gate, rebuilt_fields = _build_minus_release_fields(manifest)
    selected_artifacts = _selected_artifacts(manifest)
    artifact_release_status = selected_artifacts.get("release_status")
    release_status = selected_gate.get("release_status", "diagnostic_only")
    blockers = _as_list(selected_gate.get("blockers"))
    return {
        "kind": "minus",
        "manifest_path": str(manifest_path),
        "source_audio_path": manifest.get("raw_audio_path") or manifest.get("source_audio_path"),
        "selected_profile": manifest.get("selected_profile"),
        "selected_score": manifest.get("selected_score"),
        "release_status": release_status,
        "gate_status": selected_gate.get("status", "blocked_for_listening"),
        "artifact_release_status": artifact_release_status,
        "release_status_drift": bool(artifact_release_status is not None and artifact_release_status != release_status),
        "rebuilt_fields": rebuilt_fields,
        "blocker_reasons": _blocker_reasons(blockers),
        "blocker_labels": _blocker_labels(blockers),
        "selected_output_path": selected_artifacts.get("selected_minus_flac")
        or selected_artifacts.get("selected_instrumental_flac"),
        "listen_output_path": selected_artifacts.get("listen_selected_minus_flac")
        or selected_artifacts.get("listen_selected_instrumental_flac"),
    }


def evaluate_manifest_path(manifest_path: Path) -> dict[str, Any] | None:
    manifest = _read_manifest(manifest_path)
    mode = manifest.get("mode")
    if mode == VOCAL_MODE:
        return evaluate_vocal_manifest(manifest_path, manifest)
    if mode == MINUS_MODE:
        return evaluate_minus_manifest(manifest_path, manifest)
    return None


def iter_manifest_paths(artifact_root: Path) -> list[Path]:
    paths: list[Path] = []
    for manifest_name in SUPPORTED_MANIFEST_NAMES:
        paths.extend(artifact_root.rglob(manifest_name))
    return sorted(set(paths), key=lambda path: str(path))


def _filter_kind(item: dict[str, Any], kind: str) -> bool:
    return kind == "auto" or item.get("kind") == kind


def run_posthoc_release_evaluation(
    *,
    artifact_root: Path,
    summary_path: Path,
    kind: str = "auto",
    limit: int | None = None,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    skipped_count = 0
    if limit is not None and int(limit) <= 0:
        summary = {
            "schema_version": 1,
            "mode": "posthoc_release_evaluation",
            "artifact_root": str(artifact_root),
            "kind": kind,
            "evaluated_count": 0,
            "skipped_manifest_count": 0,
            "error_count": 0,
            "release_status_counts": {},
            "gate_status_counts": {},
            "errors": [],
            "items": [],
        }
        _write_json(summary_path, summary)
        return summary

    for manifest_path in iter_manifest_paths(artifact_root):
        try:
            item = evaluate_manifest_path(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(
                {
                    "manifest_path": str(manifest_path),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        if item is None:
            skipped_count += 1
            continue
        if not _filter_kind(item, kind):
            continue
        items.append(item)
        if limit is not None and len(items) >= max(0, int(limit)):
            break

    release_status_counts = Counter(str(item.get("release_status", "unknown")) for item in items)
    gate_status_counts = Counter(str(item.get("gate_status", "unknown")) for item in items)
    summary = {
        "schema_version": 1,
        "mode": "posthoc_release_evaluation",
        "artifact_root": str(artifact_root),
        "kind": kind,
        "evaluated_count": len(items),
        "skipped_manifest_count": skipped_count,
        "error_count": len(errors),
        "release_status_counts": dict(sorted(release_status_counts.items())),
        "gate_status_counts": dict(sorted(gate_status_counts.items())),
        "errors": errors,
        "items": items,
    }
    _write_json(summary_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize release gate status from existing adaptive audio manifests without rerunning inference.",
    )
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("summary_path", type=Path)
    parser.add_argument("--kind", choices=("auto", "vocal", "minus"), default="auto")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_posthoc_release_evaluator.py",
        entrypoint_path="/app/tools/run_posthoc_release_evaluator.py",
    )
    args = parse_args(argv)
    artifact_root = args.artifact_root.resolve()
    if not artifact_root.exists():
        raise FileNotFoundError(artifact_root)
    summary = run_posthoc_release_evaluation(
        artifact_root=artifact_root,
        summary_path=args.summary_path,
        kind=str(args.kind),
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "summary_path": str(args.summary_path),
                "evaluated_count": summary["evaluated_count"],
                "release_status_counts": summary["release_status_counts"],
                "gate_status_counts": summary["gate_status_counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
