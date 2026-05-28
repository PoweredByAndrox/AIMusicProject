from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from worker.high_quality_stem_export import (
    _read_audio_window,
    export_aggressive_melody_first_from_stem_dir,
)


def _default_output_dir(source_dir: Path, start_sec: float, end_sec: float | None) -> Path:
    def _fmt(value: float) -> str:
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return text.replace(".", "_")

    start_label = _fmt(start_sec)
    end_label = "end" if end_sec is None else _fmt(end_sec)
    return source_dir.parent / f"{source_dir.name}_preview_{start_label}_{end_label}"


def _rms(audio: np.ndarray) -> float:
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    return float(np.sqrt(np.mean(np.square(mono, dtype=np.float32))))


def _harmonic_percussive_metrics(audio: np.ndarray) -> dict[str, float]:
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    stft = librosa.stft(mono, n_fft=2048, hop_length=512)
    harmonic_stft, percussive_stft = librosa.decompose.hpss(stft, margin=(1.5, 1.5))
    harmonic = librosa.istft(harmonic_stft, hop_length=512, length=mono.shape[0]).astype(np.float32)
    percussive = librosa.istft(percussive_stft, hop_length=512, length=mono.shape[0]).astype(np.float32)
    return {
        "harmonic_rms": round(_rms(harmonic), 6),
        "percussive_rms": round(_rms(percussive), 6),
    }


def _stem_metric_summary(audio: np.ndarray, stem_name: str) -> dict[str, float]:
    summary = {"rms": round(_rms(audio), 6)}
    if stem_name in {"drums", "other"}:
        summary.update(_harmonic_percussive_metrics(audio))
    return summary


def _build_comparison_summary(
    source_dir: Path,
    output_dir: Path,
    *,
    start_sec: float,
    end_sec: float,
) -> dict[str, object]:
    compared_stems = ("vocals", "drums", "other", "melody")
    comparison: dict[str, object] = {
        "window_sec": {
            "start": float(start_sec),
            "end": float(end_sec),
        },
        "stems": {},
    }
    for stem_name in compared_stems:
        source_path = source_dir / f"{stem_name}.flac"
        output_path = output_dir / f"{stem_name}.flac"
        if not source_path.exists() or not output_path.exists():
            continue
        source_audio, _ = _read_audio_window(source_path, start_sec=start_sec, end_sec=end_sec)
        output_audio, _ = _read_audio_window(output_path, start_sec=0.0, end_sec=end_sec - start_sec)
        comparison["stems"][stem_name] = {
            "source": _stem_metric_summary(source_audio, stem_name),
            "candidate": _stem_metric_summary(output_audio, stem_name),
            "delta_rms": round(_rms(output_audio - source_audio), 6),
        }
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a preview-only HQ stem rewrite window from an existing stem directory."
    )
    parser.add_argument(
        "source_stem_dir",
        help="Existing stem export directory containing stem_manifest.json and FLAC stems.",
    )
    parser.add_argument(
        "--output-dir",
        help="Target directory for preview stems. Defaults to a sibling *_preview_<start>_<end> directory.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Preview window start in seconds. Default: 0.0",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        required=True,
        help="Preview window end in seconds.",
    )
    parser.add_argument(
        "--summary-path",
        help="Optional JSON file path to store the run summary.",
    )
    parser.add_argument("--other-move", type=float, default=0.35)
    parser.add_argument("--guitar-move", type=float, default=1.0)
    parser.add_argument("--piano-move", type=float, default=1.0)
    parser.add_argument("--drums-masked-move", type=float, default=0.08)
    parser.add_argument("--target-melody-rms", type=float, default=0.024)
    parser.add_argument("--max-melody-gain", type=float, default=1.4)
    parser.add_argument("--stage-drum-harmonic-move", type=float, default=0.35)
    parser.add_argument("--stage-harmonic-stem-percussive-move", type=float, default=0.18)
    parser.add_argument("--stage-cleanup-drum-harmonic-move", type=float, default=0.03)
    parser.add_argument("--stage-cleanup-harmonic-stem-percussive-move", type=float, default=0.25)
    parser.add_argument("--harmonic-count", type=int, default=2)
    parser.add_argument("--confidence-floor", type=float, default=0.35)
    parser.add_argument("--dsp-device", default="cuda", help="DSP device for STFT/HPSS/masking path. Default: cuda")
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="export_hq_stem_preview.py",
        entrypoint_path="/app/tools/export_hq_stem_preview.py",
    )
    args = parse_args()
    source_dir = Path(args.source_stem_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(
        source_dir,
        float(args.start_sec),
        None if args.end_sec is None else float(args.end_sec),
    )

    started_at = time.perf_counter()
    manifest = export_aggressive_melody_first_from_stem_dir(
        source_dir,
        output_dir,
        start_sec=float(args.start_sec),
        end_sec=None if args.end_sec is None else float(args.end_sec),
        other_move=float(args.other_move),
        guitar_move=float(args.guitar_move),
        piano_move=float(args.piano_move),
        drums_masked_move=float(args.drums_masked_move),
        target_melody_rms=float(args.target_melody_rms),
        max_melody_gain=float(args.max_melody_gain),
        stage_drum_harmonic_move=float(args.stage_drum_harmonic_move),
        stage_harmonic_stem_percussive_move=float(args.stage_harmonic_stem_percussive_move),
        stage_cleanup_drum_harmonic_move=float(args.stage_cleanup_drum_harmonic_move),
        stage_cleanup_harmonic_stem_percussive_move=float(args.stage_cleanup_harmonic_stem_percussive_move),
        harmonic_count=int(args.harmonic_count),
        confidence_floor=float(args.confidence_floor),
        dsp_device=str(args.dsp_device),
    )
    elapsed_sec = round(time.perf_counter() - started_at, 3)
    comparison = _build_comparison_summary(
        source_dir,
        output_dir,
        start_sec=float(args.start_sec),
        end_sec=float(args.end_sec),
    )

    summary = {
        "source_stem_dir": str(source_dir),
        "output_dir": str(output_dir),
        "manifest_path": manifest["manifest_path"],
        "preview_window_sec": {
            "start": float(args.start_sec),
            "end": None if args.end_sec is None else float(args.end_sec),
        },
        "tuning": manifest.get("tuning"),
        "dsp_device": str(args.dsp_device),
        "elapsed_sec": elapsed_sec,
        "comparison": comparison,
    }

    if args.summary_path:
        summary_path = Path(args.summary_path).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
