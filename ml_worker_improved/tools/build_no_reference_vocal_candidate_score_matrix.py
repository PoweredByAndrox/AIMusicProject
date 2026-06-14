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


MATRIX_JSON_NAME = "no_reference_vocal_candidate_score_matrix.json"
MATRIX_JSONL_NAME = "no_reference_vocal_candidate_score_matrix.jsonl"
MATRIX_SUMMARY_NAME = "no_reference_vocal_candidate_score_matrix_summary.md"

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
        raise ValueError(f"Expected JSON object at {path}")
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def parent_candidate_id(path_text: str) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    if path.name == "stem_manifest.json":
        return path.parent.name
    return path.stem


def manifest_gpu_proof(payload: dict[str, Any]) -> dict[str, Any]:
    execution_contract = payload.get("execution_contract") if isinstance(payload.get("execution_contract"), dict) else {}
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    cuda_preflight = execution_contract.get("cuda_preflight") if isinstance(execution_contract.get("cuda_preflight"), dict) else {}
    audio_separator = runtime.get("audio_separator") if isinstance(runtime.get("audio_separator"), dict) else {}
    providers = [str(provider) for provider in audio_separator.get("available_providers") or []] if isinstance(audio_separator.get("available_providers"), list) else []
    cuda_available = cuda_preflight.get("cuda_available") is True or "CUDAExecutionProvider" in providers
    gpu_required = execution_contract.get("gpu_inference_required") is True
    gpu_skipped = execution_contract.get("gpu_inference_skipped") is True
    cuda_provider_available = audio_separator.get("cuda_provider_available") is True
    return {
        "has_gpu_proof": bool((gpu_required and not gpu_skipped and cuda_available) or cuda_provider_available),
        "gpu_inference_required": gpu_required,
        "gpu_inference_skipped": gpu_skipped,
        "cuda_available": bool(cuda_available),
        "cuda_provider_available": bool(cuda_provider_available),
        "device_name": str(cuda_preflight.get("device_name") or ""),
        "available_providers": providers,
        "expected_acceleration_mode": str(execution_contract.get("expected_acceleration_mode") or ""),
    }


def read_candidate_manifest(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {
            "candidate_route_id": "",
            "model_name": "",
            "target_stem": "",
            "manifest_readable": False,
            "gpu_provenance": manifest_gpu_proof({}),
        }
    path = Path(path_text)
    if not path.exists():
        return {
            "candidate_route_id": parent_candidate_id(path_text),
            "model_name": "",
            "target_stem": "",
            "manifest_readable": False,
            "gpu_provenance": manifest_gpu_proof({}),
        }
    payload = read_json(path)
    return {
        "candidate_route_id": parent_candidate_id(path_text),
        "model_name": str(payload.get("model_name") or payload.get("model_id") or ""),
        "target_stem": str(payload.get("target_stem") or ""),
        "manifest_readable": True,
        "gpu_provenance": manifest_gpu_proof(payload),
    }


def vocal_judge_entry(row: dict[str, Any], source_path: Path) -> dict[str, Any]:
    blockers = as_list(row.get("vocal_blockers"))
    decision = str(row.get("vocal_decision") or row.get("decision") or "")
    score_mode = str(row.get("score_mode") or "")
    accepted = score_mode == "vocal" and decision == "accept_candidate" and not blockers
    return {
        "source_path": str(source_path),
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "candidate_manifest": str(row.get("candidate_manifest") or ""),
        "mix_path": str(row.get("mix_path") or row.get("source_audio_path") or ""),
        "vocal_path": str(row.get("selected_vocal_path") or row.get("vocal_path") or ""),
        "instrumental_path": str(row.get("instrumental_path") or row.get("minus_path") or ""),
        "score_mode": score_mode,
        "decision": decision,
        "accepted": accepted,
        "vocal_score": as_float(row.get("vocal_score") or row.get("overall_no_reference_score")),
        "overall_no_reference_score": as_float(row.get("overall_no_reference_score")),
        "confidence": as_float(row.get("confidence")),
        "vocal_blockers": blockers,
        "reasons": stable_unique(as_list(row.get("reasons"))),
        "vocal_dominant_defect": str(row.get("vocal_dominant_defect") or ""),
    }


def selector_entries(row: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    ranked = row.get("ranked_candidates") if isinstance(row.get("ranked_candidates"), list) else []
    selected_candidate_id = str(row.get("selected_candidate_id") or "")
    entries: list[dict[str, Any]] = []
    for ranked_row in ranked:
        if not isinstance(ranked_row, dict):
            continue
        candidate_id = str(ranked_row.get("candidate_id") or "")
        selected = bool(candidate_id and candidate_id == selected_candidate_id)
        entries.append(
            {
                "source_path": str(source_path),
                "candidate_id": candidate_id,
                "score_mode": str(row.get("score_mode") or ranked_row.get("score_mode") or ""),
                "track_id": str(row.get("track_id") or ""),
                "rank": as_int(ranked_row.get("rank")),
                "selection_score": as_float(ranked_row.get("selection_score") or ranked_row.get("vocal_score")),
                "decision": str(row.get("decision") or ""),
                "selected": selected,
                "accepted": selected and str(row.get("decision") or "") in {"accept_candidate", "select_alternative_candidate"},
            }
        )
    if selected_candidate_id and not any(entry["candidate_id"] == selected_candidate_id for entry in entries):
        entries.append(
            {
                "source_path": str(source_path),
                "candidate_id": selected_candidate_id,
                "score_mode": str(row.get("score_mode") or ""),
                "track_id": str(row.get("track_id") or ""),
                "rank": 1,
                "selection_score": 0.0,
                "decision": str(row.get("decision") or ""),
                "selected": True,
                "accepted": str(row.get("decision") or "") in {"accept_candidate", "select_alternative_candidate"},
            }
        )
    return entries


def listen_entry(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    return {
        "source_path": str(source_path),
        "candidate_id": str(payload.get("candidate_id") or ""),
        "track_id": str(payload.get("track_id") or ""),
        "selected_vocal_path": str(payload.get("selected_vocal_path") or payload.get("vocal_path") or ""),
        "delivery_allowed": payload.get("delivery_allowed") is True,
        "delivery_status": str(payload.get("delivery_status") or ""),
        "presentation_status": str(payload.get("presentation_status") or ""),
        "release_status": str(payload.get("release_status") or ""),
    }


def candidate_record_key(entry: dict[str, Any]) -> str:
    manifest_id = parent_candidate_id(str(entry.get("candidate_manifest") or ""))
    if manifest_id:
        return manifest_id
    for key in ("candidate_id", "candidate_label"):
        value = str(entry.get(key) or "")
        if value:
            return value
    vocal_path = str(entry.get("vocal_path") or entry.get("selected_vocal_path") or "")
    return parent_candidate_id(vocal_path) or vocal_path


def add_aliases(record: dict[str, Any], aliases: Iterable[str]) -> None:
    record["aliases"] = stable_unique([*record.get("aliases", []), *[str(alias) for alias in aliases if str(alias)]])


def find_record(
    records: dict[str, dict[str, Any]],
    *,
    vocal_path: str = "",
    candidate_id: str = "",
) -> dict[str, Any] | None:
    aliases = {candidate_id} if candidate_id else set()
    for record in records.values():
        if vocal_path and vocal_path == record.get("vocal_path"):
            return record
        if aliases.intersection(set(record.get("aliases", []))):
            return record
    return None


def empty_vocal_judge() -> dict[str, Any]:
    return {
        "accepted": False,
        "score_mode": "",
        "decision": "",
        "vocal_score": 0.0,
        "confidence": 0.0,
        "vocal_blockers": [],
        "reasons": [],
        "source_path": "",
    }


def empty_selector() -> dict[str, Any]:
    return {
        "accepted": False,
        "selected": False,
        "decision": "",
        "rank": 0,
        "selection_score": 0.0,
        "source_path": "",
    }


def empty_listen_preview() -> dict[str, Any]:
    return {
        "delivery_allowed": False,
        "delivery_status": "",
        "presentation_status": "",
        "release_status": "",
        "source_path": "",
    }


def build_records(
    *,
    vocal_judge_jsonls: Sequence[Path],
    selector_jsonls: Sequence[Path],
    listen_manifests: Sequence[Path],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for jsonl_path in vocal_judge_jsonls:
        for row in iter_jsonl(jsonl_path):
            vocal_judge = vocal_judge_entry(row, jsonl_path)
            key = candidate_record_key(vocal_judge)
            manifest_info = read_candidate_manifest(vocal_judge["candidate_manifest"])
            canonical = manifest_info["candidate_route_id"] or key
            record = records.setdefault(
                canonical,
                {
                    "canonical_candidate_id": canonical,
                    "track_id": vocal_judge["track_id"],
                    "vocal_path": vocal_judge["vocal_path"],
                    "instrumental_path": vocal_judge["instrumental_path"],
                    "candidate_manifest": vocal_judge["candidate_manifest"],
                    "manifest_info": manifest_info,
                    "aliases": [],
                },
            )
            record["vocal_judge"] = vocal_judge
            if vocal_judge["vocal_path"]:
                record["vocal_path"] = vocal_judge["vocal_path"]
            add_aliases(record, [canonical, vocal_judge["candidate_id"], manifest_info.get("candidate_route_id", "")])
    for selector_path in selector_jsonls:
        for row in iter_jsonl(selector_path):
            for selector in selector_entries(row, selector_path):
                record = find_record(records, candidate_id=selector["candidate_id"])
                if record is None:
                    continue
                record["selector"] = selector
                add_aliases(record, [selector["candidate_id"]])
    for listen_path in listen_manifests:
        listen = listen_entry(read_json(listen_path), listen_path)
        record = find_record(records, vocal_path=listen["selected_vocal_path"], candidate_id=listen["candidate_id"])
        if record is None:
            continue
        record["listen_preview"] = listen
        add_aliases(record, [listen["candidate_id"]])
    return records


def next_action_for_row(
    *,
    vocal_judge: dict[str, Any],
    selector: dict[str, Any],
    listen_preview: dict[str, Any],
) -> str:
    if listen_preview.get("delivery_allowed") is True and listen_preview.get("presentation_status") == "accepted_for_listen_pack":
        return "accepted_for_listen_pack_not_release"
    if vocal_judge.get("accepted") is True and selector.get("accepted") is True:
        return "candidate_ready_for_listen_preview_contract"
    if vocal_judge.get("accepted") is True:
        return "run_selector_and_listen_preview_contract"
    if not vocal_judge.get("source_path"):
        return "run_no_reference_result_judge_vocal"
    return "do_not_release_select_cleaner_vocal_or_new_route"


def selection_priority(
    *,
    vocal_judge: dict[str, Any],
    selector: dict[str, Any],
    listen_preview: dict[str, Any],
) -> int:
    if listen_preview.get("delivery_allowed") is True and listen_preview.get("presentation_status") == "accepted_for_listen_pack":
        return 100
    if vocal_judge.get("accepted") is True and selector.get("accepted") is True:
        return 92
    if vocal_judge.get("accepted") is True:
        return 82
    if vocal_judge.get("source_path"):
        return 40
    return 10


def matrix_row(record: dict[str, Any]) -> dict[str, Any]:
    vocal_judge = dict(empty_vocal_judge(), **record.get("vocal_judge", {}))
    selector = dict(empty_selector(), **record.get("selector", {}))
    listen_preview = dict(empty_listen_preview(), **record.get("listen_preview", {}))
    manifest_info = record.get("manifest_info") if isinstance(record.get("manifest_info"), dict) else {}
    gpu_provenance = manifest_info.get("gpu_provenance") if isinstance(manifest_info.get("gpu_provenance"), dict) else manifest_gpu_proof({})
    next_action = next_action_for_row(vocal_judge=vocal_judge, selector=selector, listen_preview=listen_preview)
    return {
        "canonical_candidate_id": record.get("canonical_candidate_id") or "",
        "track_id": record.get("track_id") or vocal_judge.get("track_id") or selector.get("track_id") or "",
        "aliases": sorted(record.get("aliases", [])),
        "candidate_manifest": record.get("candidate_manifest") or "",
        "model_name": manifest_info.get("model_name") or "",
        "target_stem": manifest_info.get("target_stem") or "",
        "vocal_path": record.get("vocal_path") or "",
        "instrumental_path": record.get("instrumental_path") or "",
        "gpu_provenance": gpu_provenance,
        "has_gpu_proof": gpu_provenance.get("has_gpu_proof") is True,
        "vocal_judge": vocal_judge,
        "selector": selector,
        "listen_preview": listen_preview,
        "selection_priority": selection_priority(vocal_judge=vocal_judge, selector=selector, listen_preview=listen_preview),
        "next_action": next_action,
        "delivery_allowed": listen_preview.get("delivery_allowed") is True,
        "presentation_status": listen_preview.get("presentation_status") or "not_accepted_for_listen_pack",
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
    }


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -int(row["selection_priority"]),
            -as_float(row["vocal_judge"].get("vocal_score")),
            -as_float(row["vocal_judge"].get("confidence")),
            0 if row.get("has_gpu_proof") else 1,
            str(row.get("canonical_candidate_id") or ""),
        ),
    )
    for index, row in enumerate(ordered, start=1):
        row["rank"] = index
    return ordered


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actions = Counter(str(row.get("next_action") or "") for row in rows)
    return {
        "matrix_type": "no_reference_vocal_candidate_score_matrix",
        "score_mode": "vocal",
        "candidate_count": len(rows),
        "result_accepted": sum(1 for row in rows if row["vocal_judge"].get("accepted") is True),
        "selector_accepted": sum(1 for row in rows if row["selector"].get("accepted") is True),
        "delivery_allowed": sum(1 for row in rows if row.get("delivery_allowed") is True),
        "ready_for_listen_preview_contract": actions.get("candidate_ready_for_listen_preview_contract", 0),
        "blocked_from_release": sum(1 for row in rows if row.get("delivery_allowed") is not True),
        "next_action_counts": dict(sorted(actions.items())),
        "top_candidate_id": str(rows[0].get("canonical_candidate_id") or "") if rows else "",
        "top_next_action": str(rows[0].get("next_action") or "") if rows else "",
        "safety": dict(SAFETY_FLAGS),
        "release_ready": False,
    }


def build_vocal_candidate_score_matrix(
    *,
    vocal_judge_jsonls: Sequence[Path],
    selector_jsonls: Sequence[Path],
    listen_manifests: Sequence[Path],
) -> dict[str, Any]:
    records = build_records(
        vocal_judge_jsonls=vocal_judge_jsonls,
        selector_jsonls=selector_jsonls,
        listen_manifests=listen_manifests,
    )
    rows = sort_rows([matrix_row(record) for record in records.values()])
    return {"rows": rows, "summary": build_summary(rows)}


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Vocal Candidate Score Matrix",
        "",
        f"- score_mode: {summary['score_mode']}",
        f"- candidate_count: {summary['candidate_count']}",
        f"- result_accepted: {summary['result_accepted']}",
        f"- selector_accepted: {summary['selector_accepted']}",
        f"- ready_for_listen_preview_contract: {summary['ready_for_listen_preview_contract']}",
        f"- delivery_allowed: {summary['delivery_allowed']}",
        f"- blocked_from_release: {summary['blocked_from_release']}",
        f"- top_candidate_id: {summary['top_candidate_id']}",
        f"- top_next_action: {summary['top_next_action']}",
        "",
        "## Next Actions",
        "",
    ]
    for action, count in summary["next_action_counts"].items():
        lines.append(f"- {action}: {count}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(matrix: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / MATRIX_JSON_NAME, matrix)
    write_jsonl(output_dir / MATRIX_JSONL_NAME, matrix["rows"])
    write_summary_md(output_dir / MATRIX_SUMMARY_NAME, matrix["summary"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a ranked no-reference vocal candidate score matrix from existing judge artifacts."
    )
    parser.add_argument("--vocal-judge-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--selector-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--listen-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.vocal_judge_jsonl:
        parser.error("at least one --vocal-judge-jsonl is required")
    matrix = build_vocal_candidate_score_matrix(
        vocal_judge_jsonls=args.vocal_judge_jsonl,
        selector_jsonls=args.selector_jsonl,
        listen_manifests=args.listen_manifest,
    )
    write_outputs(matrix, args.output_dir)
    print(json.dumps(matrix["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
