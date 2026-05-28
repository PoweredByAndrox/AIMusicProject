from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

import numpy as np
import soundfile as sf


def main() -> None:
    require_docker_runtime(
        tool_name="detect_vocal_window.py",
        entrypoint_path="/app/tools/detect_vocal_window.py",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("vocals_path")
    parser.add_argument("--window-sec", type=int, default=10)
    parser.add_argument("--frame-sec", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--summary-path")
    args = parser.parse_args()

    audio, sample_rate = sf.read(args.vocals_path, always_2d=True)
    mono = audio.mean(axis=1)

    frame_size = max(1, int(round(sample_rate * args.frame_sec)))
    frame_rms = []
    for start in range(0, len(mono) - frame_size + 1, frame_size):
        segment = mono[start:start + frame_size]
        frame_rms.append(float(np.sqrt(np.mean(segment ** 2))))
    frame_rms = np.asarray(frame_rms, dtype=np.float64)
    if frame_rms.size < args.window_sec:
        raise SystemExit("Not enough frames for requested window size")

    kernel = np.ones(args.window_sec, dtype=np.float64) / float(args.window_sec)
    scores = np.convolve(frame_rms, kernel, mode="valid")
    order = np.argsort(scores)[::-1]
    threshold = float(np.percentile(frame_rms, 65.0))

    top_windows = []
    for idx in order[: args.top_k]:
        idx = int(idx)
        local = frame_rms[idx:idx + args.window_sec]
        top_windows.append(
            {
                "start_sec": float(idx * args.frame_sec),
                "end_sec": float((idx + args.window_sec) * args.frame_sec),
                "mean_rms": float(scores[idx]),
                "active_fraction_over_p65": float(np.mean(local > threshold)),
                "min_rms": float(local.min()),
                "max_rms": float(local.max()),
            }
        )

    result = {
        "vocals_path": str(Path(args.vocals_path)),
        "sample_rate": int(sample_rate),
        "window_sec": int(args.window_sec),
        "frame_sec": float(args.frame_sec),
        "active_threshold_rms_p65": threshold,
        "best_window": top_windows[0],
        "top_windows": top_windows,
    }

    if args.summary_path:
        Path(args.summary_path).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
