from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_status_contract import sha256_file  # noqa: E402


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

ACCEPTABLE_CANDIDATE_STATUSES = {
    "rendered_minus_segment_switch_candidate",
    "rendered_minus_air_detail_repair_candidate",
    "existing_minus_candidate_proxy_pass",
}
JUDGE_GUIDED_LEAKAGE_REPAIR_MODE = "judge_guided_minus_vocal_leakage_repair"
DETAIL_GATE_FIELD = "minus_musical_detail_preservation_gate"
DETAIL_GATE_PASS_STATUSES = {"proxy_pass", "passed", "detail_preservation_pass"}
DETAIL_GATE_REQUIRED_METRICS = (
    "instrumental_integrity",
    "transient_preservation",
    "spectral_balance",
    "phase_stereo_stability",
    "mixture_consistency_score",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def candidate_row(candidate_manifest: dict[str, Any]) -> dict[str, Any]:
    row = candidate_manifest.get("candidate_row")
    if isinstance(row, dict):
        return row
    return {}


def minus_path_from_candidate(candidate_manifest: dict[str, Any]) -> str:
    row = candidate_row(candidate_manifest)
    return str(
        row.get("selected_minus_path")
        or row.get("instrumental_path")
        or row.get("minus_path")
        or candidate_manifest.get("rendered_minus_candidate_path")
        or ""
    )


def candidate_id_from_candidate(candidate_manifest: dict[str, Any]) -> str:
    row = candidate_row(candidate_manifest)
    return str(row.get("candidate_id") or candidate_manifest.get("candidate_id") or "")


def has_acceptable_candidate_origin(candidate_manifest: dict[str, Any]) -> bool:
    if str(candidate_manifest.get("render_status") or "") in ACCEPTABLE_CANDIDATE_STATUSES:
        return True
    row = candidate_row(candidate_manifest)
    return (
        str(candidate_manifest.get("mode") or "") == JUDGE_GUIDED_LEAKAGE_REPAIR_MODE
        and str(candidate_manifest.get("delivery_status") or "") == "candidate_only"
        and candidate_manifest.get("delivery_allowed") is False
        and str(row.get("candidate_family") or "") == JUDGE_GUIDED_LEAKAGE_REPAIR_MODE
        and str(row.get("candidate_origin") or "") == "judge_guided_projection_repair_candidate"
    )


def requires_post_render_minus_self_judge(candidate_manifest: dict[str, Any]) -> bool:
    row = candidate_row(candidate_manifest)
    return (
        candidate_manifest.get("requires_post_render_minus_self_judge") is True
        or row.get("requires_post_render_minus_self_judge") is True
    )


def candidate_claims_delivery(candidate_manifest: dict[str, Any]) -> bool:
    row = candidate_row(candidate_manifest)
    claim_keys = (
        "can_publish_as_result",
        "can_mark_release_ready",
        "can_publish_as_release",
        "delivery_allowed",
    )
    return any(candidate_manifest.get(key) is True or row.get(key) is True for key in claim_keys)


def first_result_judge_row(result_judge_jsonl: Path, candidate_id: str) -> dict[str, Any] | None:
    rows = list(iter_jsonl(result_judge_jsonl))
    for row in rows:
        if str(row.get("candidate_id") or "") == candidate_id:
            return row
    return rows[0] if rows else None


def first_selector_row(selector_jsonl: Path) -> dict[str, Any] | None:
    rows = list(iter_jsonl(selector_jsonl))
    return rows[0] if rows else None


def selector_selected_candidate(selector_row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(selector_row, dict):
        return {}
    selected = selector_row.get("selected_candidate")
    return selected if isinstance(selected, dict) else {}


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def detail_preservation_gate(result_judge_row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result_judge_row, dict):
        return None
    gate = result_judge_row.get(DETAIL_GATE_FIELD)
    return gate if isinstance(gate, dict) else None


def detail_preservation_failure_reason(gate: dict[str, Any] | None) -> str:
    if gate is None:
        return "minus_detail_preservation_gate_missing"
    blockers = gate.get("blockers")
    if isinstance(blockers, list) and blockers:
        return "minus_detail_preservation_gate_blockers"
    if str(gate.get("status") or "") not in DETAIL_GATE_PASS_STATUSES:
        return "minus_detail_preservation_gate_not_passed"
    metrics = gate.get("metrics")
    if not isinstance(metrics, dict):
        return "minus_detail_preservation_metrics_missing"
    required = gate.get("required_metrics")
    declared_metrics = [str(item) for item in required if str(item)] if isinstance(required, list) else []
    required_metrics = stable_unique(list(DETAIL_GATE_REQUIRED_METRICS) + declared_metrics)
    missing_metrics = [key for key in required_metrics if not finite_number(metrics.get(key))]
    if missing_metrics:
        return "minus_detail_preservation_metrics_missing"
    return ""


def self_judge_candidate(self_judge: dict[str, Any]) -> dict[str, Any]:
    candidate = self_judge.get("candidate")
    return candidate if isinstance(candidate, dict) else {}


def evidence_paths(
    *,
    candidate_manifest: dict[str, Any],
    self_judge_manifest: dict[str, Any],
    result_judge_row: dict[str, Any] | None,
    selector_row: dict[str, Any] | None,
) -> dict[str, str]:
    selected = selector_selected_candidate(selector_row)
    self_judge = self_judge_candidate(self_judge_manifest)
    return {
        "candidate_manifest_minus_path": minus_path_from_candidate(candidate_manifest),
        "self_judge_minus_path": str(self_judge.get("instrumental_path") or self_judge.get("minus_path") or ""),
        "result_judge_minus_path": str(
            (result_judge_row or {}).get("instrumental_path") or (result_judge_row or {}).get("minus_path") or ""
        ),
        "selector_minus_path": str(selected.get("instrumental_path") or selected.get("minus_path") or ""),
    }


def path_evidence_matches(paths: dict[str, str]) -> bool:
    expected = paths["candidate_manifest_minus_path"]
    return bool(expected) and all(path == expected for path in paths.values())


def failure_reason(
    *,
    candidate_manifest: dict[str, Any],
    self_judge_manifest: dict[str, Any],
    result_judge_row: dict[str, Any] | None,
    selector_row: dict[str, Any] | None,
    paths: dict[str, str],
    detail_gate: dict[str, Any] | None,
) -> str:
    if not has_acceptable_candidate_origin(candidate_manifest):
        return "candidate_not_rendered"
    if not requires_post_render_minus_self_judge(candidate_manifest):
        return "missing_post_render_self_judge_requirement"
    if candidate_claims_delivery(candidate_manifest):
        return "candidate_manifest_claims_delivery"
    overall = self_judge_manifest.get("overall") if isinstance(self_judge_manifest.get("overall"), dict) else {}
    if str(overall.get("status") or "") != "proxy_pass" or int(overall.get("blocker_count") or 0) != 0:
        return "post_render_self_judge_not_passed"
    if result_judge_row is None:
        return "result_judge_missing"
    if str(result_judge_row.get("score_mode") or "") != "minus":
        return "result_judge_not_minus_mode"
    if str(result_judge_row.get("decision") or "") != "accept_candidate":
        return "result_judge_not_accepted"
    if str(result_judge_row.get("minus_decision") or "accept_candidate") != "accept_candidate":
        return "result_judge_not_minus_accepted"
    if as_list(result_judge_row.get("minus_blockers")):
        return "result_judge_minus_blockers"
    expected_candidate_id = candidate_id_from_candidate(candidate_manifest)
    if str(result_judge_row.get("candidate_id") or "") != expected_candidate_id:
        return "result_judge_candidate_mismatch"
    detail_reason = detail_preservation_failure_reason(detail_gate)
    if detail_reason:
        return detail_reason
    if selector_row is None or str(selector_row.get("decision") or "") != "accept_candidate":
        return "selector_not_accepted"
    if str(selector_row.get("score_mode") or "") != "minus":
        return "selector_not_minus_mode"
    selected = selector_selected_candidate(selector_row)
    if str(selected.get("candidate_id") or "") != expected_candidate_id:
        return "selector_candidate_mismatch"
    if not path_evidence_matches(paths):
        return "evidence_path_mismatch"
    return ""


def source_ref(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() else "",
    }


def blocked_manifest(
    *,
    reason: str,
    candidate_manifest_json: Path,
    self_judge_manifest_json: Path,
    result_judge_jsonl: Path,
    selector_jsonl: Path,
    candidate_manifest: dict[str, Any],
    paths: dict[str, str],
    detail_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    row = candidate_row(candidate_manifest)
    return {
        "manifest_type": "no_reference_minus_listen_preview",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": reason,
        "stem_type": "minus",
        "judge_mode": "no_reference_minus_self_judge_and_selector",
        "score_mode": "minus",
        "selection_metric": "minus_score",
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "selected_minus_path": minus_path_from_candidate(candidate_manifest),
        DETAIL_GATE_FIELD: detail_gate or {"status": "missing"},
        "accepted_count": 0,
        "skipped_count": 1,
        "skipped_rows": [
            {
                "candidate_id": str(row.get("candidate_id") or ""),
                "skip_reason": reason,
                "evidence_paths": paths,
            }
        ],
        "source_candidate_manifest": source_ref(candidate_manifest_json),
        "source_self_judge_manifest": source_ref(self_judge_manifest_json),
        "source_result_judge_jsonl": source_ref(result_judge_jsonl),
        "source_selector_jsonl": source_ref(selector_jsonl),
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": dict(SAFETY_FLAGS),
    }


def accepted_manifest(
    *,
    candidate_manifest_json: Path,
    self_judge_manifest_json: Path,
    result_judge_jsonl: Path,
    selector_jsonl: Path,
    candidate_manifest: dict[str, Any],
    result_judge_row: dict[str, Any],
    selector_row: dict[str, Any],
    paths: dict[str, str],
    detail_gate: dict[str, Any],
) -> dict[str, Any]:
    row = candidate_row(candidate_manifest)
    minus_path = minus_path_from_candidate(candidate_manifest)
    return {
        "manifest_type": "no_reference_minus_listen_preview",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": "minus",
        "judge_mode": "no_reference_minus_self_judge_and_selector",
        "score_mode": "minus",
        "selection_metric": "minus_score",
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "mix_path": str(row.get("mix_path") or row.get("source_audio_path") or ""),
        "vocal_path": str(row.get("selected_vocal_path") or row.get("vocal_path") or ""),
        "minus_path": minus_path,
        "instrumental_path": minus_path,
        "selected_minus_path": minus_path,
        "selected_audio_path": minus_path,
        DETAIL_GATE_FIELD: detail_gate,
        "selected_artifacts": {
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "listen_minus_flac": minus_path,
        },
        "accepted_count": 1,
        "skipped_count": 0,
        "accepted_listen_preview_rows": [
            {
                "track_id": str(row.get("track_id") or ""),
                "candidate_id": str(row.get("candidate_id") or ""),
                "stem_type": "minus",
                "score_mode": "minus",
                "minus_path": minus_path,
                "minus_score": float(
                    result_judge_row.get("minus_score") or result_judge_row.get("overall_no_reference_score") or 0.0
                ),
                "overall_no_reference_score": float(result_judge_row.get("overall_no_reference_score") or 0.0),
                "confidence": float(result_judge_row.get("confidence") or selector_row.get("confidence") or 0.0),
                "reasons": stable_unique(
                    as_list(result_judge_row.get("reasons")) + as_list(selector_row.get("reasons"))
                ),
                DETAIL_GATE_FIELD: detail_gate,
                "evidence_paths": paths,
            }
        ],
        "skipped_rows": [],
        "source_candidate_manifest": source_ref(candidate_manifest_json),
        "source_self_judge_manifest": source_ref(self_judge_manifest_json),
        "source_result_judge_jsonl": source_ref(result_judge_jsonl),
        "source_selector_jsonl": source_ref(selector_jsonl),
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(
    *,
    candidate_manifest_json: Path,
    self_judge_manifest_json: Path,
    result_judge_jsonl: Path,
    selector_jsonl: Path,
) -> dict[str, Any]:
    candidate_manifest = read_json(candidate_manifest_json)
    self_judge_manifest = read_json(self_judge_manifest_json)
    candidate_id = candidate_id_from_candidate(candidate_manifest)
    result_judge_row = first_result_judge_row(result_judge_jsonl, candidate_id)
    selector_row = first_selector_row(selector_jsonl)
    paths = evidence_paths(
        candidate_manifest=candidate_manifest,
        self_judge_manifest=self_judge_manifest,
        result_judge_row=result_judge_row,
        selector_row=selector_row,
    )
    detail_gate = detail_preservation_gate(result_judge_row)
    reason = failure_reason(
        candidate_manifest=candidate_manifest,
        self_judge_manifest=self_judge_manifest,
        result_judge_row=result_judge_row,
        selector_row=selector_row,
        paths=paths,
        detail_gate=detail_gate,
    )
    if reason:
        return blocked_manifest(
            reason=reason,
            candidate_manifest_json=candidate_manifest_json,
            self_judge_manifest_json=self_judge_manifest_json,
            result_judge_jsonl=result_judge_jsonl,
            selector_jsonl=selector_jsonl,
            candidate_manifest=candidate_manifest,
            paths=paths,
            detail_gate=detail_gate,
        )
    assert result_judge_row is not None
    assert selector_row is not None
    return accepted_manifest(
        candidate_manifest_json=candidate_manifest_json,
        self_judge_manifest_json=self_judge_manifest_json,
        result_judge_jsonl=result_judge_jsonl,
        selector_jsonl=selector_jsonl,
        candidate_manifest=candidate_manifest,
        result_judge_row=result_judge_row,
        selector_row=selector_row,
        paths=paths,
        detail_gate=detail_gate,
    )


def build_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# No-Reference Minus Listen Preview Summary",
        "",
        f"- delivery_status: {manifest.get('delivery_status')}",
        f"- delivery_allowed: {str(manifest.get('delivery_allowed') is True).lower()}",
        f"- presentation_status: {manifest.get('presentation_status')}",
        f"- release_status: {manifest.get('release_status')}",
        f"- stem_type: {manifest.get('stem_type')}",
        f"- accepted_count: {manifest.get('accepted_count')}",
        f"- skipped_count: {manifest.get('skipped_count')}",
        f"- selected_minus_path: {manifest.get('selected_minus_path')}",
        f"- dominant_failure_reason: {manifest.get('dominant_failure_reason')}",
        "",
        "## Safety",
        "",
    ]
    safety = manifest.get("safety") if isinstance(manifest.get("safety"), dict) else {}
    for key in sorted(SAFETY_FLAGS):
        lines.append(f"- {key}: {str(safety.get(key) is True).lower()}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a contract-safe no-reference minus listen-preview manifest.")
    parser.add_argument("--candidate-manifest-json", type=Path, required=True)
    parser.add_argument("--self-judge-manifest-json", type=Path, required=True)
    parser.add_argument("--result-judge-jsonl", type=Path, required=True)
    parser.add_argument("--selector-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest(
        candidate_manifest_json=args.candidate_manifest_json,
        self_judge_manifest_json=args.self_judge_manifest_json,
        result_judge_jsonl=args.result_judge_jsonl,
        selector_jsonl=args.selector_jsonl,
    )
    write_json(args.output_manifest_json, manifest)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(build_summary(manifest), encoding="utf-8")
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
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
