from __future__ import annotations

import argparse
import json
import re
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


def _slug(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug or "candidate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple vocal-only preview candidates from one raw audio source.")
    parser.add_argument("audio_path", help="Source audio file path.")
    parser.add_argument("candidate_spec", help="JSON file with {'candidates': [...]} parameter sets.")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument("--device", default="cuda", help="Separation device. Default: cuda")
    parser.add_argument("--output-root", help="Directory under which candidate subdirectories will be created.")
    parser.add_argument("--summary-path", help="Optional JSON file path for the batch summary.")
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="batch_export_hq_vocal_previews.py",
        entrypoint_path="/app/tools/batch_export_hq_vocal_previews.py",
    )
    args = parse_args()
    audio_path = Path(args.audio_path).resolve()
    spec_path = Path(args.candidate_spec).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else ROOT_DIR / "lab_artifacts"

    if not spec_path.exists():
        raise SystemExit(f"Candidate spec not found: {spec_path}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    candidates = spec.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        raise SystemExit("Candidate spec must contain a non-empty 'candidates' list")

    summary: dict[str, object] = {
        "audio_path": str(audio_path),
        "candidate_spec": str(spec_path),
        "preview_window_sec": {
            "start": float(args.start_sec),
            "end": float(args.end_sec),
        },
        "device": str(args.device),
        "candidates": [],
    }

    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise SystemExit(f"Candidate #{index} is not an object")
        name = str(candidate.get("name") or f"candidate_{index}")
        output_dir = output_root / f"{audio_path.stem}_{_slug(name)}"
        started_at = time.perf_counter()
        manifest = export_vocal_preview_from_audio(
            audio_path,
            output_dir,
            candidate_mode=str(candidate.get("candidate_mode", "hybrid_vocals")),
            start_sec=float(args.start_sec),
            end_sec=float(args.end_sec),
            device=str(args.device),
            low_bs_weight=float(candidate.get("low_bs_weight", 0.75)),
            high_bs_weight=float(candidate.get("high_bs_weight", 0.25)),
            inactive_low_bs_weight=float(candidate.get("inactive_low_bs_weight", 0.97)),
            inactive_high_bs_weight=float(candidate.get("inactive_high_bs_weight", 0.9)),
            crossover_hz=float(candidate.get("crossover_hz", 3000.0)),
            activity_abs_floor_ratio=float(candidate.get("activity_abs_floor_ratio", 0.08)),
            activity_rel_floor=float(candidate.get("activity_rel_floor", 0.35)),
            activity_smoothing_frames=int(candidate.get("activity_smoothing_frames", 11)),
            vocal_sharpen_alpha=float(candidate.get("vocal_sharpen_alpha", 0.0)),
            tail_gate_strength=float(candidate.get("tail_gate_strength", 0.0)),
            tail_gate_floor_ratio=float(candidate.get("tail_gate_floor_ratio", 0.08)),
            tail_gate_smoothing_frames=int(candidate.get("tail_gate_smoothing_frames", 11)),
            subtractive_strength=float(candidate.get("subtractive_strength", 0.0)),
            subtractive_power=float(candidate.get("subtractive_power", 1.0)),
            subtractive_residual_floor=float(candidate.get("subtractive_residual_floor", 0.12)),
            subtractive_active_restore=float(candidate.get("subtractive_active_restore", 0.0)),
            subtractive_target_rms_ratio=float(candidate.get("subtractive_target_rms_ratio", 0.0)),
            subtractive_max_gain=float(candidate.get("subtractive_max_gain", 1.35)),
            hard_mask_threshold=float(candidate.get("hard_mask_threshold", 0.0)),
            hard_mask_power=float(candidate.get("hard_mask_power", 3.0)),
            hard_mask_residual_floor=float(candidate.get("hard_mask_residual_floor", 0.02)),
        )
        elapsed_sec = round(time.perf_counter() - started_at, 3)
        summary["candidates"].append(
            {
                "name": name,
                "candidate_mode": str(candidate.get("candidate_mode", "hybrid_vocals")),
                "output_dir": str(output_dir),
                "manifest_path": manifest["manifest_path"],
                "elapsed_sec": elapsed_sec,
                "tuning": manifest.get("tuning"),
            }
        )

    if args.summary_path:
        summary_path = Path(args.summary_path).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
