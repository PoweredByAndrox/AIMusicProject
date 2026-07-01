from __future__ import annotations

import argparse
import json
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
    "existing_vocal_candidate_proxy_pass",
    "rendered_vocal_judge_guided_silence_repair_candidate",
    "rendered_vocal_source_quiet_silence_repair_candidate",
    "rendered_vocal_segment_switch_candidate",
}
ACCEPTED_SELECTOR_DECISIONS = {"accept_candidate", "select_alternative_candidate"}
STEM_MIN_WINDOW_COUNT = 5


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


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
    return row if isinstance(row, dict) else {}


def candidate_safety(candidate_manifest: dict[str, Any]) -> dict[str, bool]:
    safety = dict(SAFETY_FLAGS)
    row = candidate_row(candidate_manifest)
    for source in (candidate_manifest.get("safety"), candidate_manifest.get("safety_flags"), row.get("safety"), row.get("safety_flags")):
        if not isinstance(source, dict):
            continue
        for key in safety:
            safety[key] = bool(safety[key] or source.get(key) is True)
    for key in safety:
        safety[key] = bool(safety[key] or candidate_manifest.get(key) is True or row.get(key) is True)
    return safety


def source_safety_violations(candidate_manifest: dict[str, Any]) -> list[str]:
    return [key for key, value in candidate_safety(candidate_manifest).items() if value is True]


def source_candidate_preview_only(candidate_manifest: dict[str, Any]) -> bool:
    return (
        str(candidate_manifest.get("delivery_status") or "") == "blocked"
        and candidate_manifest.get("delivery_allowed") is False
        and str(candidate_manifest.get("presentation_status") or "") == "preview_only_not_for_listen_pack"
        and str(candidate_manifest.get("release_status") or "") == "not_release_ready"
    )


def vocal_path_from_candidate(candidate_manifest: dict[str, Any]) -> str:
    row = candidate_row(candidate_manifest)
    return str(
        row.get("selected_vocal_path")
        or row.get("vocal_path")
        or candidate_manifest.get("rendered_vocal_candidate_path")
        or ""
    )


def candidate_id_from_candidate(candidate_manifest: dict[str, Any]) -> str:
    row = candidate_row(candidate_manifest)
    return str(row.get("candidate_id") or candidate_manifest.get("candidate_id") or "")


def first_vocal_judge_row(vocal_judge_jsonl: Path, candidate_id: str) -> dict[str, Any] | None:
    rows = list(iter_jsonl(vocal_judge_jsonl))
    for row in rows:
        if str(row.get("candidate_id") or "") == candidate_id:
            return row
    return rows[0] if rows else None


def first_selector_row(selector_jsonl: Path) -> dict[str, Any] | None:
    rows = list(iter_jsonl(selector_jsonl))
    return rows[0] if rows else None


def vocal_acceptance_decision(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    return str(row.get("vocal_decision") or "")


def vocal_judge_contract_failure(row: dict[str, Any]) -> str:
    if "vocal_score" not in row or row.get("vocal_score") in (None, ""):
        return "vocal_judge_missing_vocal_score"
    if "vocal_decision" not in row or row.get("vocal_decision") in (None, ""):
        return "vocal_judge_missing_vocal_decision"
    if vocal_acceptance_decision(row) != "accept_candidate":
        return "vocal_judge_not_accepted"
    if "vocal_blockers" not in row or row.get("vocal_blockers") is None:
        return "vocal_judge_missing_vocal_blockers"
    if as_list(row.get("vocal_blockers")):
        return "vocal_judge_has_blockers"
    if as_int(row.get("window_count")) < STEM_MIN_WINDOW_COUNT:
        return "vocal_judge_insufficient_window_count"
    return ""


def selector_selected_candidate(selector_row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(selector_row, dict):
        return {}
    selected = selector_row.get("selected_candidate")
    return selected if isinstance(selected, dict) else {}


def evidence_paths(
    *,
    candidate_manifest: dict[str, Any],
    vocal_judge_row: dict[str, Any] | None,
    selector_row: dict[str, Any] | None,
) -> dict[str, str]:
    selected = selector_selected_candidate(selector_row)
    return {
        "candidate_manifest_vocal_path": vocal_path_from_candidate(candidate_manifest),
        "vocal_judge_vocal_path": str((vocal_judge_row or {}).get("vocal_path") or ""),
        "selector_vocal_path": str(selected.get("selected_vocal_path") or selected.get("vocal_path") or ""),
    }


def path_evidence_matches(paths: dict[str, str]) -> bool:
    expected = paths["candidate_manifest_vocal_path"]
    return bool(expected) and all(path == expected for path in paths.values())


def failure_reason(
    *,
    candidate_manifest: dict[str, Any],
    vocal_judge_row: dict[str, Any] | None,
    selector_row: dict[str, Any] | None,
    paths: dict[str, str],
) -> str:
    if str(candidate_manifest.get("render_status") or "") not in ACCEPTABLE_CANDIDATE_STATUSES:
        return "candidate_not_rendered"
    if candidate_manifest.get("requires_post_render_vocal_judge") is not True:
        return "missing_post_render_vocal_judge_requirement"
    if candidate_manifest.get("can_publish_as_result") is True or candidate_manifest.get("can_mark_release_ready") is True:
        return "candidate_manifest_claims_delivery"
    if not source_candidate_preview_only(candidate_manifest):
        return "source_candidate_not_preview_only"
    if source_safety_violations(candidate_manifest):
        return "source_candidate_safety_violation"
    if vocal_judge_row is None:
        return "vocal_judge_missing"
    if str(vocal_judge_row.get("score_mode") or "") != "vocal":
        return "vocal_judge_not_vocal_mode"
    vocal_contract_reason = vocal_judge_contract_failure(vocal_judge_row)
    if vocal_contract_reason:
        return vocal_contract_reason
    expected_candidate_id = candidate_id_from_candidate(candidate_manifest)
    if str(vocal_judge_row.get("candidate_id") or "") != expected_candidate_id:
        return "vocal_judge_candidate_mismatch"
    if selector_row is None or str(selector_row.get("decision") or "") not in ACCEPTED_SELECTOR_DECISIONS:
        return "selector_not_accepted"
    if str(selector_row.get("score_mode") or "") != "vocal":
        return "selector_not_vocal_mode"
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
    vocal_judge_jsonl: Path,
    selector_jsonl: Path,
    candidate_manifest: dict[str, Any],
    paths: dict[str, str],
) -> dict[str, Any]:
    row = candidate_row(candidate_manifest)
    return {
        "manifest_type": "no_reference_vocal_listen_preview",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": reason,
        "stem_type": "vocal",
        "judge_mode": "no_reference_vocal_mode_judge_and_selector",
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "selected_vocal_path": vocal_path_from_candidate(candidate_manifest),
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
        "source_vocal_judge_jsonl": source_ref(vocal_judge_jsonl),
        "source_selector_jsonl": source_ref(selector_jsonl),
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": candidate_safety(candidate_manifest),
    }


def accepted_manifest(
    *,
    candidate_manifest_json: Path,
    vocal_judge_jsonl: Path,
    selector_jsonl: Path,
    candidate_manifest: dict[str, Any],
    vocal_judge_row: dict[str, Any],
    selector_row: dict[str, Any],
    paths: dict[str, str],
) -> dict[str, Any]:
    row = candidate_row(candidate_manifest)
    vocal_path = vocal_path_from_candidate(candidate_manifest)
    reasons = stable_unique(as_list(vocal_judge_row.get("reasons")) + as_list(selector_row.get("reasons")))
    return {
        "manifest_type": "no_reference_vocal_listen_preview",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": "vocal",
        "judge_mode": "no_reference_vocal_mode_judge_and_selector",
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "mix_path": str(row.get("mix_path") or row.get("source_audio_path") or ""),
        "vocal_path": vocal_path,
        "minus_path": str(row.get("selected_minus_path") or row.get("minus_path") or row.get("instrumental_path") or ""),
        "instrumental_path": str(row.get("instrumental_path") or row.get("minus_path") or ""),
        "selected_vocal_path": vocal_path,
        "selected_audio_path": vocal_path,
        "selected_artifacts": {
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "listen_vocals_flac": vocal_path,
            "listen_selected_vocals_flac": vocal_path,
        },
        "accepted_count": 1,
        "skipped_count": 0,
        "accepted_listen_preview_rows": [
            {
                "track_id": str(row.get("track_id") or ""),
                "candidate_id": str(row.get("candidate_id") or ""),
                "stem_type": "vocal",
                "vocal_path": vocal_path,
                "vocal_score": float(vocal_judge_row.get("vocal_score") or 0.0),
                "vocal_decision": str(vocal_judge_row.get("vocal_decision") or ""),
                "vocal_blockers": as_list(vocal_judge_row.get("vocal_blockers")),
                "window_count": as_int(vocal_judge_row.get("window_count")),
                "overall_no_reference_score": float(vocal_judge_row.get("overall_no_reference_score") or 0.0),
                "confidence": float(vocal_judge_row.get("confidence") or selector_row.get("confidence") or 0.0),
                "reasons": reasons,
                "evidence_paths": paths,
            }
        ],
        "skipped_rows": [],
        "source_candidate_manifest": source_ref(candidate_manifest_json),
        "source_vocal_judge_jsonl": source_ref(vocal_judge_jsonl),
        "source_selector_jsonl": source_ref(selector_jsonl),
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": candidate_safety(candidate_manifest),
    }


def build_manifest(
    *,
    candidate_manifest_json: Path,
    vocal_judge_jsonl: Path,
    selector_jsonl: Path,
) -> dict[str, Any]:
    candidate_manifest = read_json(candidate_manifest_json)
    candidate_id = candidate_id_from_candidate(candidate_manifest)
    vocal_judge_row = first_vocal_judge_row(vocal_judge_jsonl, candidate_id)
    selector_row = first_selector_row(selector_jsonl)
    paths = evidence_paths(
        candidate_manifest=candidate_manifest,
        vocal_judge_row=vocal_judge_row,
        selector_row=selector_row,
    )
    reason = failure_reason(
        candidate_manifest=candidate_manifest,
        vocal_judge_row=vocal_judge_row,
        selector_row=selector_row,
        paths=paths,
    )
    if reason:
        return blocked_manifest(
            reason=reason,
            candidate_manifest_json=candidate_manifest_json,
            vocal_judge_jsonl=vocal_judge_jsonl,
            selector_jsonl=selector_jsonl,
            candidate_manifest=candidate_manifest,
            paths=paths,
        )
    assert vocal_judge_row is not None
    assert selector_row is not None
    return accepted_manifest(
        candidate_manifest_json=candidate_manifest_json,
        vocal_judge_jsonl=vocal_judge_jsonl,
        selector_jsonl=selector_jsonl,
        candidate_manifest=candidate_manifest,
        vocal_judge_row=vocal_judge_row,
        selector_row=selector_row,
        paths=paths,
    )


def build_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# No-Reference Vocal Listen Preview Summary",
        "",
        f"- delivery_status: {manifest.get('delivery_status')}",
        f"- delivery_allowed: {str(manifest.get('delivery_allowed') is True).lower()}",
        f"- presentation_status: {manifest.get('presentation_status')}",
        f"- release_status: {manifest.get('release_status')}",
        f"- stem_type: {manifest.get('stem_type')}",
        f"- accepted_count: {manifest.get('accepted_count')}",
        f"- skipped_count: {manifest.get('skipped_count')}",
        f"- selected_vocal_path: {manifest.get('selected_vocal_path')}",
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
    parser = argparse.ArgumentParser(description="Build a contract-safe no-reference vocal listen-preview manifest.")
    parser.add_argument("--candidate-manifest-json", type=Path, required=True)
    parser.add_argument("--vocal-judge-jsonl", type=Path, required=True)
    parser.add_argument("--selector-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = build_manifest(
        candidate_manifest_json=args.candidate_manifest_json,
        vocal_judge_jsonl=args.vocal_judge_jsonl,
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
