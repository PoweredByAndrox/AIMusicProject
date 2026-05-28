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
from export_hq_stem_preview import _build_comparison_summary
from worker.high_quality_stem_export import export_aggressive_melody_first_from_stem_dir


def _slug(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug or "candidate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple HQ stem preview candidates from one source stem directory."
    )
    parser.add_argument("source_stem_dir", help="Existing stem export directory containing FLAC stems.")
    parser.add_argument("candidate_spec", help="JSON file with {'candidates': [...]} parameter sets.")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument(
        "--output-root",
        help="Directory under which candidate subdirectories will be created. Defaults to the source parent.",
    )
    parser.add_argument("--summary-path", help="Optional JSON file path for the batch summary.")
    parser.add_argument("--dsp-device", default="cuda", help="DSP device for STFT/HPSS/masking path. Default: cuda")
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="batch_export_hq_stem_previews.py",
        entrypoint_path="/app/tools/batch_export_hq_stem_previews.py",
    )
    args = parse_args()
    source_dir = Path(args.source_stem_dir).resolve()
    spec_path = Path(args.candidate_spec).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else source_dir.parent

    if not spec_path.exists():
        raise SystemExit(f"Candidate spec not found: {spec_path}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    candidates = spec.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        raise SystemExit("Candidate spec must contain a non-empty 'candidates' list")

    summary: dict[str, object] = {
        "source_stem_dir": str(source_dir),
        "candidate_spec": str(spec_path),
        "preview_window_sec": {
            "start": float(args.start_sec),
            "end": float(args.end_sec),
        },
        "dsp_device": str(args.dsp_device),
        "candidates": [],
    }

    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise SystemExit(f"Candidate #{index} is not an object")
        name = str(candidate.get("name") or f"candidate_{index}")
        output_dir = output_root / f"{source_dir.name}_{_slug(name)}"

        started_at = time.perf_counter()
        manifest = export_aggressive_melody_first_from_stem_dir(
            source_dir,
            output_dir,
            start_sec=float(args.start_sec),
            end_sec=float(args.end_sec),
            other_move=float(candidate.get("other_move", 0.35)),
            guitar_move=float(candidate.get("guitar_move", 1.0)),
            piano_move=float(candidate.get("piano_move", 1.0)),
            drums_masked_move=float(candidate.get("drums_masked_move", 0.08)),
            target_melody_rms=float(candidate.get("target_melody_rms", 0.024)),
            max_melody_gain=float(candidate.get("max_melody_gain", 1.4)),
            stage_drum_harmonic_move=float(candidate.get("stage_drum_harmonic_move", 0.35)),
            stage_harmonic_stem_percussive_move=float(candidate.get("stage_harmonic_stem_percussive_move", 0.18)),
            stage_cleanup_drum_harmonic_move=float(candidate.get("stage_cleanup_drum_harmonic_move", 0.03)),
            stage_cleanup_harmonic_stem_percussive_move=float(candidate.get("stage_cleanup_harmonic_stem_percussive_move", 0.25)),
            harmonic_count=int(candidate.get("harmonic_count", 2)),
            confidence_floor=float(candidate.get("confidence_floor", 0.35)),
            dsp_device=str(args.dsp_device),
        )
        elapsed_sec = round(time.perf_counter() - started_at, 3)
        comparison = _build_comparison_summary(
            source_dir,
            output_dir,
            start_sec=float(args.start_sec),
            end_sec=float(args.end_sec),
        )
        summary["candidates"].append(
            {
                "name": name,
                "output_dir": str(output_dir),
                "manifest_path": manifest["manifest_path"],
                "elapsed_sec": elapsed_sec,
                "tuning": manifest.get("tuning"),
                "comparison": comparison,
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
