from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.run_no_reference_result_judge import (  # noqa: E402
    as_list,
    judge_audio,
    resample_audio,
    resolve_path,
    stable_unique,
)
from tools.no_reference_score_matrix import pair_compatibility_score, score_candidate  # noqa: E402


BASE_FORBIDDEN_ACTIONS = [
    "do_not_require_reference_stems",
    "do_not_require_reference_vocal_minus",
    "do_not_require_human_csv",
    "do_not_start_finetune",
    "do_not_train",
    "do_not_add_models",
    "do_not_add_cleanup_debleed_denoise",
    "do_not_relax_gates",
    "do_not_publish_preview_as_final_result",
    "do_not_mark_release_ready",
    "do_not_export_additional_audio",
]

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "can_publish_as_result": False,
    "can_mark_release_ready": False,
}

NOOP_RMS_THRESHOLD = 5.0e-5
NOOP_PEAK_THRESHOLD = 1.0e-4
IMPROVED_DELTA_THRESHOLD = 0.005
REGRESSED_DELTA_THRESHOLD = -0.005


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_stem(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"vocal", "vocals"}:
        return "vocal"
    if text in {"minus", "instrumental"}:
        return "minus"
    if text == "both":
        return "both"
    return "unknown"


def target_stems(source_stem: str) -> list[str]:
    if source_stem == "both":
        return ["vocal", "minus"]
    if source_stem in {"vocal", "minus"}:
        return [source_stem]
    return []


def read_window(path_text: str, start_sec: float, duration_sec: float) -> tuple[np.ndarray, int]:
    path = resolve_path(path_text)
    info = sf.info(str(path))
    start = max(0, int(round(start_sec * info.samplerate)))
    frames = max(1, int(round(duration_sec * info.samplerate)))
    audio, sample_rate = sf.read(str(path), start=start, frames=frames, always_2d=True, dtype="float32")
    if audio.size == 0:
        raise ValueError(f"Empty target window: {path_text}")
    return audio.astype(np.float32), int(sample_rate)


def align_to_mix(audio: np.ndarray, source_rate: int, mix_rate: int, target_frames: int) -> np.ndarray:
    aligned = resample_audio(audio, source_rate, mix_rate)
    if aligned.shape[0] > target_frames:
        return aligned[:target_frames]
    if aligned.shape[0] < target_frames:
        pad = np.zeros((target_frames - aligned.shape[0], aligned.shape[1]), dtype=np.float32)
        return np.concatenate([aligned, pad], axis=0)
    return aligned


def window_metrics(
    *,
    mix_path: str,
    vocal_path: str,
    minus_path: str,
    start_sec: float,
    duration_sec: float,
) -> dict[str, Any]:
    mix, mix_rate = read_window(mix_path, start_sec, duration_sec)
    vocal, vocal_rate = read_window(vocal_path, start_sec, duration_sec)
    minus, minus_rate = read_window(minus_path, start_sec, duration_sec)
    vocal = align_to_mix(vocal, vocal_rate, mix_rate, mix.shape[0])
    minus = align_to_mix(minus, minus_rate, mix_rate, mix.shape[0])
    return judge_audio(mix, vocal, minus, mix_rate)


def window_difference(base_path: str, preview_path: str, start_sec: float, duration_sec: float) -> dict[str, float | bool]:
    base, base_rate = read_window(base_path, start_sec, duration_sec)
    preview, preview_rate = read_window(preview_path, start_sec, duration_sec)
    preview = align_to_mix(preview, preview_rate, base_rate, base.shape[0])
    if preview.shape[1] != base.shape[1]:
        channels = min(preview.shape[1], base.shape[1])
        base = base[:, :channels]
        preview = preview[:, :channels]
    diff = preview - base
    rms = float(np.sqrt(np.mean(np.square(diff))) + 1.0e-12)
    peak = float(np.max(np.abs(diff))) if diff.size else 0.0
    return {
        "rms_diff": round(rms, 9),
        "peak_diff": round(peak, 9),
        "differs": bool(rms > NOOP_RMS_THRESHOLD or peak > NOOP_PEAK_THRESHOLD),
    }


def blocker_count(metrics: dict[str, Any], source_stem: str = "both") -> int:
    source_stem = normalize_stem(source_stem)
    if source_stem == "vocal":
        reasons = as_list(score_candidate(metrics, "vocal").get("vocal_blockers"))
    elif source_stem == "minus":
        reasons = as_list(score_candidate(metrics, "minus").get("minus_blockers"))
    else:
        reasons = as_list(pair_compatibility_score(metrics, metrics).get("pair_blockers"))
        reasons.extend([reason for reason in as_list(metrics.get("reasons")) if reason != "no_reference_proxy_passed"])
    return len(set(reasons))


def score(metrics: dict[str, Any], source_stem: str = "both") -> float:
    source_stem = normalize_stem(source_stem)
    if source_stem == "vocal":
        return round(as_float(score_candidate(metrics, "vocal").get("vocal_score")), 6)
    if source_stem == "minus":
        return round(as_float(score_candidate(metrics, "minus").get("minus_score")), 6)
    return round(as_float(pair_compatibility_score(metrics, metrics).get("pair_compatibility_score")), 6)


def classify_preview_decision(
    *,
    base_alt_paths_identical: bool,
    preview_differs_from_base: bool,
    score_delta: float,
    blocker_delta: int,
) -> tuple[str, str]:
    if base_alt_paths_identical or not preview_differs_from_base:
        return "preview_noop_self_blend", "fix_candidate_identity/path-distinctness"
    if blocker_delta > 0 or score_delta <= REGRESSED_DELTA_THRESHOLD:
        return "preview_regressed_window", "discard_preview_and_fix_planner"
    if score_delta >= IMPROVED_DELTA_THRESHOLD and blocker_delta <= 0:
        return "preview_improved_window", "expand_preview_policy"
    return "preview_inconclusive", "keep_preview_for_diagnostics_only"


def row_key(row: dict[str, Any]) -> tuple[str, float, float, str, str, str]:
    return (
        str(row.get("track_id") or ""),
        as_float(row.get("window_start_sec")),
        as_float(row.get("window_duration_sec")),
        normalize_stem(row.get("source_stem")),
        str(row.get("base_candidate_id") or ""),
        str(row.get("alternate_candidate_id") or ""),
    )


def find_contract_row(preview_row: dict[str, Any], contract_rows: list[dict[str, Any]], dry_run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in contract_rows if str(row.get("render_status") or "") == "render_ready"] or dry_run_rows
    preview_key = row_key(preview_row)
    for row in rows:
        if row_key(row) == preview_key:
            return row
    for row in rows:
        if (
            str(row.get("track_id") or "") == str(preview_row.get("track_id") or "")
            and as_float(row.get("window_start_sec")) == as_float(preview_row.get("window_start_sec"))
            and as_float(row.get("window_duration_sec")) == as_float(preview_row.get("window_duration_sec"))
        ):
            return row
    raise ValueError(f"No matching render contract row for preview row: {preview_row}")


def find_mix_path(contract_row: dict[str, Any], windowed_rows: list[dict[str, Any]]) -> str:
    track_id = str(contract_row.get("track_id") or "")
    base_vocal = str(contract_row.get("base_vocal_path") or "")
    base_minus = str(contract_row.get("base_minus_path") or "")
    for row in windowed_rows:
        if (
            str(row.get("track_id") or "") == track_id
            and str(row.get("vocal_path") or "") == base_vocal
            and str(row.get("instrumental_path") or row.get("minus_path") or "") == base_minus
            and row.get("mix_path")
        ):
            return str(row["mix_path"])
    for row in windowed_rows:
        if str(row.get("track_id") or "") == track_id and row.get("mix_path"):
            return str(row["mix_path"])
    raise ValueError(f"No mix_path found in windowed judge rows for track_id={track_id}")


def path_same(left: Any, right: Any) -> bool:
    return str(left or "") == str(right or "")


def build_row(preview_row: dict[str, Any], contract_row: dict[str, Any], windowed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    start_sec = as_float(preview_row.get("window_start_sec"))
    duration_sec = as_float(preview_row.get("window_duration_sec"))
    source_stem = normalize_stem(preview_row.get("source_stem"))
    mix_path = find_mix_path(contract_row, windowed_rows)
    base_vocal_path = str(contract_row.get("base_vocal_path") or "")
    base_minus_path = str(contract_row.get("base_minus_path") or "")
    alt_vocal_path = str(contract_row.get("alt_vocal_path") or "")
    alt_minus_path = str(contract_row.get("alt_minus_path") or "")
    preview_vocal_path = str(preview_row.get("rendered_vocal_preview_path") or "")
    preview_minus_path = str(preview_row.get("rendered_minus_preview_path") or "")
    base_metrics = window_metrics(
        mix_path=mix_path,
        vocal_path=base_vocal_path,
        minus_path=base_minus_path,
        start_sec=start_sec,
        duration_sec=duration_sec,
    )
    preview_metrics = window_metrics(
        mix_path=mix_path,
        vocal_path=preview_vocal_path,
        minus_path=preview_minus_path,
        start_sec=start_sec,
        duration_sec=duration_sec,
    )
    base_alt_same_vocal = path_same(base_vocal_path, alt_vocal_path)
    base_alt_same_minus = path_same(base_minus_path, alt_minus_path)
    active_stems = target_stems(source_stem)
    stem_diffs: dict[str, dict[str, float | bool]] = {}
    if "vocal" in active_stems:
        stem_diffs["vocal"] = window_difference(base_vocal_path, preview_vocal_path, start_sec, duration_sec)
    if "minus" in active_stems:
        stem_diffs["minus"] = window_difference(base_minus_path, preview_minus_path, start_sec, duration_sec)
    preview_differs = any(bool(value.get("differs")) for value in stem_diffs.values())
    base_alt_paths_identical = bool(active_stems) and all(
        (stem == "vocal" and base_alt_same_vocal) or (stem == "minus" and base_alt_same_minus)
        for stem in active_stems
    )
    base_score = score(base_metrics, source_stem)
    preview_score = score(preview_metrics, source_stem)
    score_delta = round(preview_score - base_score, 6)
    blockers_delta = blocker_count(preview_metrics, source_stem) - blocker_count(base_metrics, source_stem)
    decision, allowed_next_step = classify_preview_decision(
        base_alt_paths_identical=base_alt_paths_identical,
        preview_differs_from_base=preview_differs,
        score_delta=score_delta,
        blocker_delta=blockers_delta,
    )
    return {
        "track_id": str(preview_row.get("track_id") or ""),
        "preview_vocal_path": preview_vocal_path,
        "preview_minus_path": preview_minus_path,
        "base_vocal_path": base_vocal_path,
        "base_minus_path": base_minus_path,
        "alt_vocal_path": alt_vocal_path,
        "alt_minus_path": alt_minus_path,
        "base_candidate_id": str(preview_row.get("base_candidate_id") or ""),
        "alternate_candidate_id": str(preview_row.get("alternate_candidate_id") or ""),
        "base_alt_same_candidate_id": path_same(preview_row.get("base_candidate_id"), preview_row.get("alternate_candidate_id")),
        "base_alt_same_vocal_path": base_alt_same_vocal,
        "base_alt_same_minus_path": base_alt_same_minus,
        "preview_differs_from_base": preview_differs,
        "window_start_sec": round(start_sec, 6),
        "window_duration_sec": round(duration_sec, 6),
        "base_window_score": base_score,
        "preview_window_score": preview_score,
        "score_delta": score_delta,
        "blocker_delta": blockers_delta,
        "preview_decision": decision,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "allowed_next_step": allowed_next_step,
        "base_window_reasons": as_list(base_metrics.get("reasons")),
        "preview_window_reasons": as_list(preview_metrics.get("reasons")),
        "target_window_diff": stem_diffs,
        "mix_path": mix_path,
        "source_stem": source_stem,
        "forbidden_actions": stable_unique(BASE_FORBIDDEN_ACTIONS + as_list(preview_row.get("forbidden_actions")) + as_list(contract_row.get("forbidden_actions"))),
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = Counter(str(row.get("preview_decision") or "") for row in rows)
    next_steps = Counter(str(row.get("allowed_next_step") or "") for row in rows)
    average_delta = round(sum(as_float(row.get("score_delta")) for row in rows) / len(rows), 6) if rows else 0.0
    return {
        "preview_rows_judged": len(rows),
        "preview_noop_self_blend": decisions.get("preview_noop_self_blend", 0),
        "preview_improved_window": decisions.get("preview_improved_window", 0),
        "preview_regressed_window": decisions.get("preview_regressed_window", 0),
        "preview_inconclusive": decisions.get("preview_inconclusive", 0),
        "base_alt_same_candidate_id_count": sum(1 for row in rows if row.get("base_alt_same_candidate_id") is True),
        "base_alt_same_vocal_path_count": sum(1 for row in rows if row.get("base_alt_same_vocal_path") is True),
        "base_alt_same_minus_path_count": sum(1 for row in rows if row.get("base_alt_same_minus_path") is True),
        "preview_differs_from_base_count": sum(1 for row in rows if row.get("preview_differs_from_base") is True),
        "average_score_delta": average_delta,
        "allowed_next_step_counts": dict(sorted(next_steps.items())),
        "can_publish_as_result_count": sum(1 for row in rows if row.get("can_publish_as_result") is True),
        "can_mark_release_ready_count": sum(1 for row in rows if row.get("can_mark_release_ready") is True),
        "safety": dict(SAFETY_FLAGS),
    }


def build_preview_window_judge(
    preview_manifest: dict[str, Any],
    render_contract_rows: list[dict[str, Any]],
    dry_run_rows: list[dict[str, Any]],
    windowed_judge_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    preview_rows = preview_manifest.get("rows") or []
    if not isinstance(preview_rows, list):
        raise ValueError("preview manifest rows must be a list")
    rows = [
        build_row(preview_row, find_contract_row(preview_row, render_contract_rows, dry_run_rows), windowed_judge_rows)
        for preview_row in sorted([row for row in preview_rows if isinstance(row, dict)], key=row_key)
    ]
    return {"rows": rows, "summary": build_summary(rows)}


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# No-Reference Segment Blend Preview Window Judge Summary",
        "",
        f"- preview rows judged: {summary['preview_rows_judged']}",
        f"- preview_noop_self_blend: {summary['preview_noop_self_blend']}",
        f"- preview_improved_window: {summary['preview_improved_window']}",
        f"- preview_regressed_window: {summary['preview_regressed_window']}",
        f"- preview_inconclusive: {summary['preview_inconclusive']}",
        f"- base_alt_same_candidate_id count: {summary['base_alt_same_candidate_id_count']}",
        f"- base_alt_same_vocal_path count: {summary['base_alt_same_vocal_path_count']}",
        f"- base_alt_same_minus_path count: {summary['base_alt_same_minus_path_count']}",
        f"- preview_differs_from_base count: {summary['preview_differs_from_base_count']}",
        f"- average_score_delta: {summary['average_score_delta']}",
        f"- can_publish_as_result_count: {summary['can_publish_as_result_count']}",
        f"- can_mark_release_ready_count: {summary['can_mark_release_ready_count']}",
        "",
        "## Allowed Next Steps",
        "",
    ]
    for key, value in summary["allowed_next_step_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], *, output_jsonl: Path, summary_md: Path) -> None:
    write_jsonl(output_jsonl, result["rows"])
    write_summary_md(summary_md, result["summary"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge a no-reference segment blend preview on its exact target window.")
    parser.add_argument("--preview-manifest-json", type=Path, required=True)
    parser.add_argument("--render-contract-jsonl", type=Path, required=True)
    parser.add_argument("--dry-run-jsonl", type=Path, required=True)
    parser.add_argument("--windowed-judge-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_preview_window_judge(
        read_json(args.preview_manifest_json),
        list(iter_jsonl(args.render_contract_jsonl)),
        list(iter_jsonl(args.dry_run_jsonl)),
        list(iter_jsonl(args.windowed_judge_jsonl)),
    )
    write_outputs(result, output_jsonl=args.output_jsonl, summary_md=args.summary_md)
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
