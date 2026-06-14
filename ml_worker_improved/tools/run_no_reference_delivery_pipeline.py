from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import soundfile as sf


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_status_contract import ArtifactContractError, can_promote_to_listen_result  # noqa: E402
from tools.build_no_reference_delivery_gate import (  # noqa: E402
    build_delivery_gate_manifest,
    build_summary as build_gate_summary,
    write_summary_md as write_gate_summary_md,
)
from tools.build_no_reference_listen_preview_manifest import (  # noqa: E402
    SAFETY_FLAGS,
    build_manifest as build_listen_preview_manifest,
    build_summary as build_listen_preview_summary,
    is_accepted_optimizer_row,
    iter_jsonl,
    listen_row_from_optimizer,
    write_json,
    write_summary_md as write_listen_preview_summary_md,
)
from tools.run_no_reference_phase_alignment_optimizer import (  # noqa: E402
    DEFAULT_MAX_OPPOSITE_REGRESSION,
    DEFAULT_MIN_DELTA,
    DEFAULT_PHASE_TRIGGER,
    build_phase_alignment_result,
    write_outputs as write_phase_alignment_outputs,
)
from tools.run_no_reference_windowed_judge import main as windowed_judge_main  # noqa: E402


MODES = ("vocal", "minus", "both")
DURATION_TOLERANCE_SEC = 0.25


def log_event(phase: str, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "phase": phase,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_inference_skipped": True,
        "gpu_inference_skipped_reason": "no_reference_delivery_pipeline_uses_existing_candidates_and_cpu_dsp_scoring",
    }
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


START_TIME = time.monotonic()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def accepted_delivery_candidates(optimizer_result_jsonl: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(optimizer_result_jsonl))
    candidates = [listen_row_from_optimizer(row, optimizer_result_jsonl) for row in rows if is_accepted_optimizer_row(row)]
    return [row for row in candidates if has_required_dense_judge_fields(row)]


def has_required_dense_judge_fields(row: dict[str, Any]) -> bool:
    return all(
        [
            str(row.get("track_id") or "").strip(),
            str(row.get("candidate_id") or "").strip(),
            str(row.get("mix_path") or "").strip(),
            str(row.get("vocal_path") or "").strip(),
            str(row.get("instrumental_path") or row.get("minus_path") or "").strip(),
        ]
    )


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "candidate_rows": output_dir / "no_reference_delivery_candidates.jsonl",
        "rejected_candidates": output_dir / "no_reference_delivery_rejected_candidates.jsonl",
        "cache": output_dir / "no_reference_metric_cache.jsonl",
        "dense_vocal": output_dir / "no_reference_dense_vocal.jsonl",
        "dense_vocal_summary": output_dir / "no_reference_dense_vocal_summary.md",
        "dense_minus": output_dir / "no_reference_dense_minus.jsonl",
        "dense_minus_summary": output_dir / "no_reference_dense_minus_summary.md",
        "dense_both": output_dir / "no_reference_dense_both.jsonl",
        "dense_both_summary": output_dir / "no_reference_dense_both_summary.md",
        "dense_gate_manifest": output_dir / "no_reference_dense_delivery_gate_manifest.json",
        "dense_gate_summary": output_dir / "no_reference_dense_delivery_gate_summary.md",
        "listen_preview_manifest": output_dir / "no_reference_listen_preview_manifest.json",
        "listen_preview_summary": output_dir / "no_reference_listen_preview_summary.md",
        "pipeline_manifest": output_dir / "no_reference_delivery_pipeline_manifest.json",
        "pipeline_summary": output_dir / "no_reference_delivery_pipeline_summary.md",
        "phase_rescue_dir": output_dir / "phase_alignment_rescue",
    }


def phase_rescue_paths(output_dir: Path) -> dict[str, Path]:
    rescue_dir = output_dir / "phase_alignment_rescue"
    return {
        "candidate_rows": rescue_dir / "no_reference_delivery_candidates.jsonl",
        "cache": rescue_dir / "no_reference_metric_cache.jsonl",
        "dense_vocal": rescue_dir / "no_reference_dense_vocal.jsonl",
        "dense_vocal_summary": rescue_dir / "no_reference_dense_vocal_summary.md",
        "dense_minus": rescue_dir / "no_reference_dense_minus.jsonl",
        "dense_minus_summary": rescue_dir / "no_reference_dense_minus_summary.md",
        "dense_both": rescue_dir / "no_reference_dense_both.jsonl",
        "dense_both_summary": rescue_dir / "no_reference_dense_both_summary.md",
        "dense_gate_manifest": rescue_dir / "no_reference_dense_delivery_gate_manifest.json",
        "dense_gate_summary": rescue_dir / "no_reference_dense_delivery_gate_summary.md",
        "listen_preview_manifest": rescue_dir / "no_reference_listen_preview_manifest.json",
        "listen_preview_summary": rescue_dir / "no_reference_listen_preview_summary.md",
        "phase_optimizer_result": rescue_dir / "no_reference_phase_alignment_optimizer_result.jsonl",
        "phase_optimizer_summary": rescue_dir / "no_reference_phase_alignment_optimizer_summary.md",
    }


def audio_duration_sec(path: str) -> float:
    info = sf.info(path)
    return round(float(info.frames) / float(info.samplerate), 3)


def preflight_candidate(row: dict[str, Any], *, duration_tolerance_sec: float = DURATION_TOLERANCE_SEC) -> tuple[bool, dict[str, Any] | None]:
    def rejected_base(skip_reason: str) -> dict[str, Any]:
        rejected = dict(row)
        rejected.update(
            {
                "skip_reason": skip_reason,
                "delivery_allowed": False,
                "delivery_status": "rejected",
                "presentation_status": "blocked_from_listen_pack",
                "release_status": "listen_preview_only",
                "can_mark_release_ready": False,
                "can_publish_as_release": False,
                "audio_copied": False,
                "audio_exported": False,
            }
        )
        return rejected

    try:
        mix_duration = audio_duration_sec(str(row.get("mix_path") or ""))
        vocal_duration = audio_duration_sec(str(row.get("vocal_path") or ""))
        minus_duration = audio_duration_sec(str(row.get("instrumental_path") or row.get("minus_path") or ""))
    except Exception as exc:
        rejected = rejected_base("candidate_audio_unreadable")
        rejected.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return False, rejected

    max_delta = max(abs(mix_duration - vocal_duration), abs(mix_duration - minus_duration))
    if max_delta > duration_tolerance_sec:
        rejected = rejected_base("candidate_duration_mismatch")
        rejected.update(
            {
                "mix_duration_sec": mix_duration,
                "vocal_duration_sec": vocal_duration,
                "minus_duration_sec": minus_duration,
                "duration_tolerance_sec": duration_tolerance_sec,
                "max_duration_delta_sec": round(max_delta, 3),
            }
        )
        return False, rejected
    return True, None


def preflight_delivery_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in candidates:
        ok, rejected_row = preflight_candidate(row)
        if ok:
            accepted.append(row)
        elif rejected_row is not None:
            rejected.append(rejected_row)
    duration_mismatch_count = sum(1 for row in rejected if row.get("skip_reason") == "candidate_duration_mismatch")
    return accepted, rejected, {
        "input_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "duration_mismatch_count": duration_mismatch_count,
        "audio_unreadable_count": sum(1 for row in rejected if row.get("skip_reason") == "candidate_audio_unreadable"),
        "duration_tolerance_sec": DURATION_TOLERANCE_SEC,
        "rejected_candidates_jsonl": "",
    }


def run_windowed_judges(
    *,
    candidate_rows_jsonl: Path,
    paths: dict[str, Path],
    window_seconds: float,
    window_step_seconds: float,
    max_windows_per_candidate: int,
    max_candidates: int,
) -> None:
    for mode in MODES:
        log_event("dense_windowed_judge", "start", score_mode=mode)
        output_jsonl = paths[f"dense_{mode}"]
        summary_md = paths[f"dense_{mode}_summary"]
        windowed_judge_main(
            [
                "--candidates-jsonl",
                str(candidate_rows_jsonl),
                "--output-jsonl",
                str(output_jsonl),
                "--summary-md",
                str(summary_md),
                "--cache-jsonl",
                str(paths["cache"]),
                "--window-seconds",
                str(window_seconds),
                "--window-step-seconds",
                str(window_step_seconds),
                "--max-windows-per-candidate",
                str(max_windows_per_candidate),
                "--max-candidates",
                str(max_candidates),
                "--score-mode",
                mode,
            ]
        )
        log_event("dense_windowed_judge", "end", score_mode=mode, output_jsonl=str(output_jsonl), summary_md=str(summary_md))


def build_dense_delivery_outputs(
    *,
    optimizer_result_jsonl: Path,
    paths: dict[str, Path],
    min_window_count: int,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    log_event("dense_delivery_gate", "start", optimizer_result_jsonl=str(optimizer_result_jsonl))
    gate_manifest = build_delivery_gate_manifest(
        optimizer_result_jsonl=optimizer_result_jsonl,
        vocal_judge_jsonl=paths["dense_vocal"],
        minus_judge_jsonl=paths["dense_minus"],
        both_judge_jsonl=paths["dense_both"],
        min_window_count=min_window_count,
    )
    write_json(paths["dense_gate_manifest"], gate_manifest)
    write_gate_summary_md(paths["dense_gate_summary"], build_gate_summary(gate_manifest))
    log_event(
        "dense_delivery_gate",
        "end",
        accepted_count=gate_manifest.get("accepted_count"),
        skipped_count=gate_manifest.get("skipped_count"),
        delivery_allowed=gate_manifest.get("delivery_allowed"),
    )

    log_event("listen_preview_manifest", "start", optimizer_result_jsonl=str(optimizer_result_jsonl))
    listen_manifest = build_listen_preview_manifest(
        optimizer_result_jsonl,
        dense_delivery_gate_manifest_json=paths["dense_gate_manifest"],
    )
    write_json(paths["listen_preview_manifest"], listen_manifest)
    write_listen_preview_summary_md(paths["listen_preview_summary"], build_listen_preview_summary(listen_manifest))
    try:
        final_promotable = can_promote_to_listen_result(paths["listen_preview_manifest"])
    except ArtifactContractError:
        final_promotable = False
    log_event(
        "listen_preview_manifest",
        "end",
        accepted_count=listen_manifest.get("accepted_count"),
        delivery_allowed=listen_manifest.get("delivery_allowed"),
        final_promotable=final_promotable,
    )
    return gate_manifest, listen_manifest, final_promotable


def run_phase_alignment_rescue(
    *,
    candidates: list[dict[str, Any]],
    output_dir: Path,
    initial_paths: dict[str, Path],
    window_seconds: float,
    window_step_seconds: float,
    max_windows_per_candidate: int,
    max_candidates: int,
    min_window_count: int,
) -> dict[str, Any]:
    paths = phase_rescue_paths(output_dir)
    log_event("phase_alignment_rescue", "start", candidate_rows=len(candidates), output_dir=str(paths["phase_optimizer_result"].parent))
    windowed_rows = [
        row
        for path in (initial_paths["dense_vocal"], initial_paths["dense_minus"], initial_paths["dense_both"])
        for row in iter_jsonl(path)
    ]
    phase_result = build_phase_alignment_result(
        candidates,
        output_dir=paths["phase_optimizer_result"].parent,
        min_delta=DEFAULT_MIN_DELTA,
        phase_trigger=DEFAULT_PHASE_TRIGGER,
        max_opposite_regression=DEFAULT_MAX_OPPOSITE_REGRESSION,
        max_candidates=max_candidates,
        windowed_judge_rows=windowed_rows,
        emit_progress=True,
    )
    write_phase_alignment_outputs(phase_result, paths["phase_optimizer_result"].parent)
    selected_count = int(phase_result["summary"].get("phase_corrected_candidate_selected") or 0)
    log_event("phase_alignment_rescue", "phase_candidates_built", selected_count=selected_count)
    if selected_count <= 0:
        write_jsonl(paths["candidate_rows"], [])
        return {
            "attempted": True,
            "selected_count": selected_count,
            "accepted_delivery_candidate_rows": 0,
            "rescue_promotable": False,
            "optimizer_result_jsonl": str(paths["phase_optimizer_result"]),
            "listen_preview_manifest": "",
            "reason": "no_phase_corrected_delivery_candidate",
        }

    phase_candidates = accepted_delivery_candidates(paths["phase_optimizer_result"])
    write_jsonl(paths["candidate_rows"], phase_candidates)
    if not phase_candidates:
        return {
            "attempted": True,
            "selected_count": selected_count,
            "accepted_delivery_candidate_rows": 0,
            "rescue_promotable": False,
            "optimizer_result_jsonl": str(paths["phase_optimizer_result"]),
            "listen_preview_manifest": "",
            "reason": "no_phase_corrected_delivery_candidate",
        }

    run_windowed_judges(
        candidate_rows_jsonl=paths["candidate_rows"],
        paths=paths,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        max_windows_per_candidate=max_windows_per_candidate,
        max_candidates=max_candidates,
    )
    _gate_manifest, _listen_manifest, rescue_promotable = build_dense_delivery_outputs(
        optimizer_result_jsonl=paths["phase_optimizer_result"],
        paths=paths,
        min_window_count=min_window_count,
    )
    return {
        "attempted": True,
        "selected_count": selected_count,
        "accepted_delivery_candidate_rows": len(phase_candidates),
        "rescue_promotable": rescue_promotable,
        "optimizer_result_jsonl": str(paths["phase_optimizer_result"]),
        "listen_preview_manifest": str(paths["listen_preview_manifest"]),
        "dense_delivery_gate_manifest": str(paths["dense_gate_manifest"]),
        "reason": "" if rescue_promotable else "phase_corrected_candidate_failed_dense_gate",
    }


def promote_rescue_listen_preview(*, rescue_paths: dict[str, Path], final_paths: dict[str, Path]) -> None:
    write_json(final_paths["listen_preview_manifest"], json.loads(rescue_paths["listen_preview_manifest"].read_text(encoding="utf-8")))
    write_text(final_paths["listen_preview_summary"], rescue_paths["listen_preview_summary"].read_text(encoding="utf-8"))


def build_pipeline_manifest(
    *,
    optimizer_result_jsonl: Path,
    output_dir: Path,
    paths: dict[str, Path],
    accepted_candidate_count: int,
    final_promotable: bool,
    phase_alignment_rescue: dict[str, Any] | None = None,
    candidate_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_payload = json.loads(paths["listen_preview_manifest"].read_text(encoding="utf-8"))
    return {
        "manifest_type": "no_reference_delivery_pipeline",
        "optimizer_result_jsonl": str(optimizer_result_jsonl),
        "output_dir": str(output_dir),
        "accepted_delivery_candidate_rows": int(accepted_candidate_count),
        "final_status": "accepted_for_listen_pack" if final_promotable else "blocked_from_listen_pack",
        "delivery_allowed": bool(final_payload.get("delivery_allowed")),
        "presentation_status": str(final_payload.get("presentation_status") or ""),
        "release_status": str(final_payload.get("release_status") or ""),
        "can_mark_release_ready": bool(final_payload.get("can_mark_release_ready")),
        "audio_copied": False,
        "audio_exported": False,
        "artifacts": {key: str(value) for key, value in sorted(paths.items())},
        "candidate_preflight": candidate_preflight or {},
        "phase_alignment_rescue": phase_alignment_rescue or {"attempted": False},
        "safety": dict(SAFETY_FLAGS),
    }


def write_pipeline_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Delivery Pipeline Summary",
        "",
        f"- final_status: {manifest['final_status']}",
        f"- delivery_allowed: {str(manifest['delivery_allowed']).lower()}",
        f"- presentation_status: {manifest['presentation_status']}",
        f"- release_status: {manifest['release_status']}",
        f"- can_mark_release_ready: {str(manifest['can_mark_release_ready']).lower()}",
        f"- accepted_delivery_candidate_rows: {manifest['accepted_delivery_candidate_rows']}",
        f"- candidate_preflight_rejected_count: {manifest.get('candidate_preflight', {}).get('rejected_count', 0)}",
        f"- candidate_preflight_duration_mismatch_count: {manifest.get('candidate_preflight', {}).get('duration_mismatch_count', 0)}",
        f"- audio_copied: {str(manifest['audio_copied']).lower()}",
        f"- audio_exported: {str(manifest['audio_exported']).lower()}",
        f"- phase_alignment_rescue_attempted: {str(manifest['phase_alignment_rescue'].get('attempted') is True).lower()}",
        f"- phase_alignment_rescue_promotable: {str(manifest['phase_alignment_rescue'].get('rescue_promotable') is True).lower()}",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in manifest["artifacts"].items():
        lines.append(f"- {key}: {value}")
    write_text(path, "\n".join(lines) + "\n")


def run_pipeline(
    *,
    optimizer_result_jsonl: Path,
    output_dir: Path,
    window_seconds: float,
    window_step_seconds: float,
    max_windows_per_candidate: int,
    max_candidates: int,
    min_window_count: int,
    enable_phase_alignment_rescue: bool = False,
) -> dict[str, Any]:
    paths = output_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_event("prepare_candidates", "start", optimizer_result_jsonl=str(optimizer_result_jsonl), output_dir=str(output_dir))
    raw_candidates = accepted_delivery_candidates(optimizer_result_jsonl)
    candidates, rejected_candidates, candidate_preflight = preflight_delivery_candidates(raw_candidates)
    candidate_preflight["rejected_candidates_jsonl"] = str(paths["rejected_candidates"])
    write_jsonl(paths["candidate_rows"], candidates)
    write_jsonl(paths["rejected_candidates"], rejected_candidates)
    log_event(
        "prepare_candidates",
        "end",
        candidate_rows=len(candidates),
        rejected_candidate_rows=len(rejected_candidates),
        candidate_rows_jsonl=str(paths["candidate_rows"]),
        rejected_candidates_jsonl=str(paths["rejected_candidates"]),
    )

    run_windowed_judges(
        candidate_rows_jsonl=paths["candidate_rows"],
        paths=paths,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        max_windows_per_candidate=max_windows_per_candidate,
        max_candidates=max_candidates,
    )

    _gate_manifest, _listen_manifest, final_promotable = build_dense_delivery_outputs(
        optimizer_result_jsonl=optimizer_result_jsonl,
        paths=paths,
        min_window_count=min_window_count,
    )
    phase_alignment_rescue = {"attempted": False}
    if not final_promotable and enable_phase_alignment_rescue and candidates:
        phase_alignment_rescue = run_phase_alignment_rescue(
            candidates=candidates,
            output_dir=output_dir,
            initial_paths=paths,
            window_seconds=window_seconds,
            window_step_seconds=window_step_seconds,
            max_windows_per_candidate=max_windows_per_candidate,
            max_candidates=max_candidates,
            min_window_count=min_window_count,
        )
        if phase_alignment_rescue.get("rescue_promotable") is True:
            promote_rescue_listen_preview(rescue_paths=phase_rescue_paths(output_dir), final_paths=paths)
            final_promotable = True

    pipeline_manifest = build_pipeline_manifest(
        optimizer_result_jsonl=optimizer_result_jsonl,
        output_dir=output_dir,
        paths=paths,
        accepted_candidate_count=len(candidates),
        final_promotable=final_promotable,
        phase_alignment_rescue=phase_alignment_rescue,
        candidate_preflight=candidate_preflight,
    )
    write_json(paths["pipeline_manifest"], pipeline_manifest)
    write_pipeline_summary(paths["pipeline_summary"], pipeline_manifest)
    return pipeline_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the no-reference dense-gated listen-preview delivery pipeline.")
    parser.add_argument("--optimizer-result-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=6.0)
    parser.add_argument("--window-step-seconds", type=float, default=6.0)
    parser.add_argument("--max-windows-per-candidate", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--min-window-count", type=int, default=5)
    parser.add_argument("--enable-phase-alignment-rescue", action="store_true")
    args = parser.parse_args(argv)

    manifest = run_pipeline(
        optimizer_result_jsonl=args.optimizer_result_jsonl,
        output_dir=args.output_dir,
        window_seconds=args.window_seconds,
        window_step_seconds=args.window_step_seconds,
        max_windows_per_candidate=args.max_windows_per_candidate,
        max_candidates=args.max_candidates,
        min_window_count=args.min_window_count,
        enable_phase_alignment_rescue=args.enable_phase_alignment_rescue,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["final_status"] == "accepted_for_listen_pack" else 1


if __name__ == "__main__":
    raise SystemExit(main())
