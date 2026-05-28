from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from worker.high_quality_stem_export import export_vocal_preview_from_audio


def _default_output_dir(audio_path: Path, candidate_mode: str, start_sec: float, end_sec: float | None) -> Path:
    def _fmt(value: float) -> str:
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return text.replace(".", "_")

    start_label = _fmt(start_sec)
    end_label = "end" if end_sec is None else _fmt(end_sec)
    return ROOT_DIR / "lab_artifacts" / f"{audio_path.stem}_{candidate_mode}_{start_label}_{end_label}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a vocal-only preview candidate from raw audio.")
    parser.add_argument("audio_path", help="Source audio file path.")
    parser.add_argument(
        "--candidate-mode",
        choices=("ft_vocals", "bs_vocals", "hybrid_vocals"),
        default="hybrid_vocals",
    )
    parser.add_argument("--output-dir", help="Target directory for vocals.flac and stem_manifest.json")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument("--device", default="cuda", help="Separation device. Default: cuda")
    parser.add_argument("--summary-path", help="Optional JSON file path for run summary")
    parser.add_argument("--low-bs-weight", type=float, default=0.75)
    parser.add_argument("--high-bs-weight", type=float, default=0.25)
    parser.add_argument("--inactive-low-bs-weight", type=float, default=0.97)
    parser.add_argument("--inactive-high-bs-weight", type=float, default=0.9)
    parser.add_argument("--crossover-hz", type=float, default=3000.0)
    parser.add_argument("--activity-abs-floor-ratio", type=float, default=0.08)
    parser.add_argument("--activity-rel-floor", type=float, default=0.35)
    parser.add_argument("--activity-smoothing-frames", type=int, default=11)
    parser.add_argument("--vocal-sharpen-alpha", type=float, default=0.0)
    parser.add_argument("--tail-gate-strength", type=float, default=0.0)
    parser.add_argument("--tail-gate-floor-ratio", type=float, default=0.08)
    parser.add_argument("--tail-gate-smoothing-frames", type=int, default=11)
    parser.add_argument("--subtractive-strength", type=float, default=0.0)
    parser.add_argument("--subtractive-power", type=float, default=1.0)
    parser.add_argument("--subtractive-residual-floor", type=float, default=0.12)
    parser.add_argument("--subtractive-active-restore", type=float, default=0.0)
    parser.add_argument("--subtractive-target-rms-ratio", type=float, default=0.0)
    parser.add_argument("--subtractive-max-gain", type=float, default=1.35)
    parser.add_argument("--hard-mask-threshold", type=float, default=0.0)
    parser.add_argument("--hard-mask-power", type=float, default=3.0)
    parser.add_argument("--hard-mask-residual-floor", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="export_hq_vocal_preview.py",
        entrypoint_path="/app/tools/export_hq_vocal_preview.py",
    )
    args = parse_args()
    audio_path = Path(args.audio_path).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(
        audio_path,
        str(args.candidate_mode),
        float(args.start_sec),
        None if args.end_sec is None else float(args.end_sec),
    )

    started_at = time.perf_counter()
    manifest = export_vocal_preview_from_audio(
        audio_path,
        output_dir,
        candidate_mode=str(args.candidate_mode),
        start_sec=float(args.start_sec),
        end_sec=None if args.end_sec is None else float(args.end_sec),
        device=str(args.device),
        low_bs_weight=float(args.low_bs_weight),
        high_bs_weight=float(args.high_bs_weight),
        inactive_low_bs_weight=float(args.inactive_low_bs_weight),
        inactive_high_bs_weight=float(args.inactive_high_bs_weight),
        crossover_hz=float(args.crossover_hz),
        activity_abs_floor_ratio=float(args.activity_abs_floor_ratio),
        activity_rel_floor=float(args.activity_rel_floor),
        activity_smoothing_frames=int(args.activity_smoothing_frames),
        vocal_sharpen_alpha=float(args.vocal_sharpen_alpha),
        tail_gate_strength=float(args.tail_gate_strength),
        tail_gate_floor_ratio=float(args.tail_gate_floor_ratio),
        tail_gate_smoothing_frames=int(args.tail_gate_smoothing_frames),
        subtractive_strength=float(args.subtractive_strength),
        subtractive_power=float(args.subtractive_power),
        subtractive_residual_floor=float(args.subtractive_residual_floor),
        subtractive_active_restore=float(args.subtractive_active_restore),
        subtractive_target_rms_ratio=float(args.subtractive_target_rms_ratio),
        subtractive_max_gain=float(args.subtractive_max_gain),
        hard_mask_threshold=float(args.hard_mask_threshold),
        hard_mask_power=float(args.hard_mask_power),
        hard_mask_residual_floor=float(args.hard_mask_residual_floor),
    )
    elapsed_sec = round(time.perf_counter() - started_at, 3)
    summary = {
        "audio_path": str(audio_path),
        "output_dir": str(output_dir),
        "manifest_path": manifest["manifest_path"],
        "candidate_mode": str(args.candidate_mode),
        "preview_window_sec": {
            "start": float(args.start_sec),
            "end": None if args.end_sec is None else float(args.end_sec),
        },
        "device": str(args.device),
        "elapsed_sec": elapsed_sec,
        "tuning": manifest.get("tuning"),
    }
    if args.summary_path:
        summary_path = Path(args.summary_path).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
