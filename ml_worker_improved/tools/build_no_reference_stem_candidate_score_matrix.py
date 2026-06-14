from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_status_contract import sha256_file, status_view_from_manifest  # noqa: E402


MATRIX_JSON_NAME = "no_reference_stem_candidate_score_matrix.json"
MATRIX_JSONL_NAME = "no_reference_stem_candidate_score_matrix.jsonl"
MATRIX_SUMMARY_NAME = "no_reference_stem_candidate_score_matrix_summary.md"

SUPPORTED_STEMS = {"melody", "drums", "bass", "other"}

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


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def source_ref(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() else "",
    }


def nested_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def safety_from_manifest(payload: dict[str, Any]) -> dict[str, bool]:
    safety = dict(SAFETY_FLAGS)
    for source in (payload, nested_dict(payload, "safety"), nested_dict(payload, "safety_flags")):
        for key in safety:
            safety[key] = bool(safety[key] or source.get(key) is True)
    return safety


def safety_violation(safety: dict[str, bool]) -> str:
    for key in sorted(safety):
        if safety.get(key) is True:
            return key
    return ""


def gate_passed(payload: dict[str, Any], key: str) -> bool:
    return str(nested_dict(payload, key).get("status") or "") == "passed"


def selected_audio_exists(payload: dict[str, Any]) -> bool:
    path = str(payload.get("selected_audio_path") or payload.get("selected_stem_path") or "")
    return bool(path) and Path(path).exists()


def candidate_key(stem_type: str, selected_audio_path: str) -> tuple[str, str]:
    return (str(stem_type or "").strip().lower(), str(selected_audio_path or "").replace("\\", "/"))


def load_semantic_judge_rows(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = candidate_key(str(row.get("stem_type") or ""), str(row.get("selected_audio_path") or ""))
        rows[key] = row
    return rows


def has_gpu_proof(payload: dict[str, Any]) -> bool:
    gate = nested_dict(payload, "gpu_provenance_gate")
    return (
        str(gate.get("status") or "") == "passed"
        and gate.get("gpu_inference_required") is True
        and gate.get("gpu_inference_skipped") is False
        and gate.get("cuda_available") is True
        and str(gate.get("resolved_device") or "").lower() == "cuda"
    )


def has_cpu_dsp_provenance(payload: dict[str, Any]) -> bool:
    gate = nested_dict(payload, "gpu_provenance_gate")
    return (
        str(gate.get("status") or "") == "passed"
        and gate.get("cpu_dsp_only_route") is True
        and gate.get("gpu_inference_required") is False
        and gate.get("gpu_inference_skipped") is True
        and str(gate.get("expected_acceleration_mode") or "").startswith("cpu_dsp")
    )


def has_acceleration_provenance(payload: dict[str, Any]) -> bool:
    return has_gpu_proof(payload) or has_cpu_dsp_provenance(payload)


def audio_proxy_components(payload: dict[str, Any]) -> dict[str, float]:
    gate = nested_dict(payload, "audio_integrity_gate")
    duration = as_float(gate.get("duration_sec"))
    rms = as_float(gate.get("rms"))
    peak = as_float(gate.get("peak"))
    min_rms = max(as_float(gate.get("min_rms"), 1e-4), 1e-8)
    min_peak = max(as_float(gate.get("min_peak"), 1e-4), 1e-8)
    max_peak = max(as_float(gate.get("max_peak"), 1.01), min_peak)
    duration_score = clip01(duration / 30.0)
    presence_score = clip01(rms / (min_rms * 100.0))
    peak_presence_score = clip01(peak / (min_peak * 100.0))
    if peak <= max_peak:
        headroom_score = 1.0
    else:
        headroom_score = clip01(1.0 - ((peak - max_peak) / max_peak))
    return {
        "duration_score": round(duration_score, 6),
        "presence_score": round(presence_score, 6),
        "peak_presence_score": round(peak_presence_score, 6),
        "headroom_score": round(headroom_score, 6),
    }


def selection_score(payload: dict[str, Any], *, eligible: bool, semantic_score: float | None = None) -> float:
    if not eligible:
        return 0.0
    components = audio_proxy_components(payload)
    proxy_score = (
        0.35
        + 0.20
        + 0.15 * components["duration_score"]
        + 0.15 * components["presence_score"]
        + 0.10 * components["peak_presence_score"]
        + 0.05 * components["headroom_score"]
    )
    if semantic_score is None:
        score = proxy_score
    else:
        score = 0.65 * proxy_score + 0.35 * clip01(semantic_score)
    return round(clip01(score), 6)


def next_action(
    *,
    stem_type: str,
    listenable: bool,
    audio_ok: bool,
    acceleration_ok: bool,
    source_ok: bool,
    path_ok: bool,
    safety_reason: str,
    semantic_blocked: bool,
) -> str:
    if stem_type not in SUPPORTED_STEMS:
        return "do_not_release_unsupported_stem_type"
    if safety_reason:
        return "do_not_release_source_safety_violation"
    if not listenable:
        return "do_not_release_generate_better_stem_candidate"
    if not (audio_ok and acceleration_ok and source_ok and path_ok):
        return "do_not_release_missing_audio_gpu_or_source_gate"
    if semantic_blocked:
        return "do_not_release_semantic_judge_blocked"
    return "accepted_for_listen_pack_not_release"


def matrix_row(path: Path, semantic_rows: dict[tuple[str, str], dict[str, Any]] | None = None) -> dict[str, Any]:
    payload = read_json(path)
    view = status_view_from_manifest(path)
    stem_type = view.stem_type
    selected_audio_path = str(payload.get("selected_audio_path") or payload.get("selected_stem_path") or "")
    semantic_row = (semantic_rows or {}).get(candidate_key(stem_type, selected_audio_path))
    semantic_decision = str((semantic_row or {}).get("decision") or "")
    semantic_score = as_float((semantic_row or {}).get("stem_semantic_score"), default=0.0) if semantic_row else None
    semantic_blockers = (semantic_row or {}).get("stem_semantic_blockers") if semantic_row else []
    if not isinstance(semantic_blockers, list):
        semantic_blockers = [str(semantic_blockers)]
    semantic_verified = bool((semantic_row or {}).get("semantic_quality_verified") is True)
    semantic_blocked = bool(semantic_row and semantic_decision != "accept_candidate")
    audio_ok = gate_passed(payload, "audio_integrity_gate")
    gpu_ok = has_gpu_proof(payload)
    acceleration_ok = has_acceleration_provenance(payload)
    source_ok = gate_passed(payload, "source_contract_gate")
    path_ok = selected_audio_exists(payload)
    safety = safety_from_manifest(payload)
    safety_reason = safety_violation(safety)
    action = next_action(
        stem_type=stem_type,
        listenable=view.listenable,
        audio_ok=audio_ok,
        acceleration_ok=acceleration_ok,
        source_ok=source_ok,
        path_ok=path_ok,
        safety_reason=safety_reason,
        semantic_blocked=semantic_blocked,
    )
    eligible = action == "accepted_for_listen_pack_not_release"
    audio_gate = nested_dict(payload, "audio_integrity_gate")
    return {
        "matrix_type": "no_reference_stem_candidate_score_matrix",
        "score_mode": "stem",
        "stem_type": stem_type,
        "track_id": str(payload.get("track_id") or ""),
        "model_name": str(payload.get("model_name") or ""),
        "selected_audio_path": selected_audio_path,
        "delivery_status": view.delivery_status,
        "delivery_allowed": bool(view.delivery_allowed and eligible),
        "presentation_status": view.presentation_status,
        "release_status": view.release_status,
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "semantic_quality_verified": semantic_verified,
        "semantic_quality_status": "proxy_semantic_pass" if semantic_verified else "not_verified_no_reference_proxy_matrix_only",
        "semantic_judge": {
            "available": semantic_row is not None,
            "decision": semantic_decision,
            "accepted": semantic_decision == "accept_candidate",
        },
        "stem_semantic_score": semantic_score if semantic_score is not None else 0.0,
        "stem_semantic_blockers": semantic_blockers,
        "has_audio_integrity_gate": audio_ok,
        "has_gpu_proof": gpu_ok,
        "has_acceleration_provenance": acceleration_ok,
        "has_source_contract_gate": source_ok,
        "selected_audio_exists": path_ok,
        "source_safety_violation": safety_reason,
        "audio_duration_sec": as_float(audio_gate.get("duration_sec")),
        "audio_rms": as_float(audio_gate.get("rms")),
        "audio_peak": as_float(audio_gate.get("peak")),
        "audio_proxy_components": audio_proxy_components(payload),
        "selection_score": selection_score(payload, eligible=eligible, semantic_score=semantic_score),
        "selection_priority": 100 if eligible else 0,
        "next_action": action,
        "source_listen_manifest": source_ref(path),
        "safety": safety,
    }


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("selection_priority") or 0),
            float(row.get("selection_score") or 0.0),
            str(row.get("stem_type") or ""),
        ),
        reverse=True,
    )


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("next_action") == "accepted_for_listen_pack_not_release"]
    unsupported = [row for row in rows if row.get("next_action") == "do_not_release_unsupported_stem_type"]
    gate_blocked = [row for row in rows if row.get("next_action") == "do_not_release_missing_audio_gpu_or_source_gate"]
    semantic_blocked = [row for row in rows if row.get("next_action") == "do_not_release_semantic_judge_blocked"]
    safety = dict(SAFETY_FLAGS)
    for row in rows:
        row_safety = row.get("safety")
        if not isinstance(row_safety, dict):
            continue
        for key in safety:
            safety[key] = bool(safety[key] or row_safety.get(key) is True)
    return {
        "matrix_type": "no_reference_stem_candidate_score_matrix",
        "candidate_count": len(rows),
        "accepted_listen_preview_count": len(accepted),
        "unsupported_stem_count": len(unsupported),
        "gate_blocked_count": len(gate_blocked),
        "semantic_judge_blocked_count": len(semantic_blocked),
        "semantic_quality_verified_count": sum(1 for row in rows if row.get("semantic_quality_verified") is True),
        "release_ready": False,
        "semantic_quality_verified": bool(rows) and all(row.get("semantic_quality_verified") is True for row in rows),
        "top_candidate_stem": str(rows[0].get("stem_type") or "") if rows else "",
        "top_candidate_score": float(rows[0].get("selection_score") or 0.0) if rows else 0.0,
        "safety": safety,
    }


def build_stem_candidate_score_matrix(
    listen_manifest_paths: Sequence[Path],
    *,
    semantic_judge_jsonl: Path | None = None,
) -> dict[str, Any]:
    semantic_rows = load_semantic_judge_rows(semantic_judge_jsonl)
    rows = sort_rows([matrix_row(path, semantic_rows) for path in listen_manifest_paths])
    return {
        "matrix_type": "no_reference_stem_candidate_score_matrix",
        "score_mode": "stem",
        "ranked_candidates": rows,
        "summary": build_summary(rows),
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "source_semantic_judge_jsonl": source_ref(semantic_judge_jsonl) if semantic_judge_jsonl is not None else {},
    }


def discover_listen_manifests(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("no_reference_stem_listen_preview_manifest.json"))


def write_summary_md(path: Path, matrix: dict[str, Any]) -> None:
    summary = matrix.get("summary") if isinstance(matrix.get("summary"), dict) else {}
    lines = [
        "# No-Reference Stem Candidate Score Matrix",
        "",
        f"- matrix_type: {matrix.get('matrix_type')}",
        f"- score_mode: {matrix.get('score_mode')}",
        f"- candidate_count: {summary.get('candidate_count')}",
        f"- accepted_listen_preview_count: {summary.get('accepted_listen_preview_count')}",
        f"- unsupported_stem_count: {summary.get('unsupported_stem_count')}",
        f"- gate_blocked_count: {summary.get('gate_blocked_count')}",
        f"- semantic_judge_blocked_count: {summary.get('semantic_judge_blocked_count')}",
        f"- semantic_quality_verified_count: {summary.get('semantic_quality_verified_count')}",
        f"- release_ready: {str(summary.get('release_ready') is True).lower()}",
        f"- semantic_quality_verified: {str(summary.get('semantic_quality_verified') is True).lower()}",
        "",
        "## Ranked Candidates",
        "",
    ]
    for row in matrix.get("ranked_candidates", []):
        lines.append(
            "- "
            f"{row.get('stem_type')} "
            f"score={row.get('selection_score')} "
            f"action={row.get('next_action')} "
            f"path={row.get('selected_audio_path')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(output_root: Path, matrix: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / MATRIX_JSON_NAME, matrix)
    write_jsonl(output_root / MATRIX_JSONL_NAME, matrix.get("ranked_candidates", []))
    write_summary_md(output_root / MATRIX_SUMMARY_NAME, matrix)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a no-reference non-vocal stem candidate score matrix.")
    parser.add_argument("--listen-manifest-json", type=Path, action="append", default=[])
    parser.add_argument("--listen-manifest-root", type=Path)
    parser.add_argument("--semantic-judge-jsonl", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)

    manifests = list(args.listen_manifest_json)
    if args.listen_manifest_root is not None:
        manifests.extend(discover_listen_manifests(args.listen_manifest_root))
    manifests = sorted(dict.fromkeys(manifests))
    matrix = build_stem_candidate_score_matrix(manifests, semantic_judge_jsonl=args.semantic_judge_jsonl)
    write_outputs(args.output_root, matrix)
    print(
        json.dumps(
            {
                "candidate_count": matrix["summary"]["candidate_count"],
                "accepted_listen_preview_count": matrix["summary"]["accepted_listen_preview_count"],
                "unsupported_stem_count": matrix["summary"]["unsupported_stem_count"],
                "gate_blocked_count": matrix["summary"]["gate_blocked_count"],
                "semantic_judge_blocked_count": matrix["summary"]["semantic_judge_blocked_count"],
                "semantic_quality_verified_count": matrix["summary"]["semantic_quality_verified_count"],
                "release_ready": matrix["summary"]["release_ready"],
                "top_candidate_stem": matrix["summary"]["top_candidate_stem"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
