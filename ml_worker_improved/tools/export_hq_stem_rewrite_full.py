from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from worker.high_quality_stem_export import export_aggressive_melody_first_from_stem_dir


def _emit(message: str) -> None:
    print(f"[hq_stem_rewrite_full] {message}", flush=True)


def _copy_listen_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def _write_flac_copy(source_audio_path: Path, target_flac_path: Path) -> None:
    if source_audio_path.suffix.lower() == ".flac":
        shutil.copyfile(source_audio_path, target_flac_path)
        return
    audio, sample_rate = sf.read(str(source_audio_path), always_2d=True)
    sf.write(str(target_flac_path), audio, sample_rate, subtype="PCM_24")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite an existing HQ stem directory with stronger melody-first cleanup over the full track."
    )
    parser.add_argument("source_stem_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--other-move", type=float, default=0.45)
    parser.add_argument("--guitar-move", type=float, default=1.0)
    parser.add_argument("--piano-move", type=float, default=1.0)
    parser.add_argument("--drums-masked-move", type=float, default=0.12)
    parser.add_argument("--target-melody-rms", type=float, default=0.026)
    parser.add_argument("--max-melody-gain", type=float, default=1.45)
    parser.add_argument("--stage-drum-harmonic-move", type=float, default=0.42)
    parser.add_argument("--stage-harmonic-stem-percussive-move", type=float, default=0.22)
    parser.add_argument("--stage-cleanup-drum-harmonic-move", type=float, default=0.05)
    parser.add_argument("--stage-cleanup-harmonic-stem-percussive-move", type=float, default=0.28)
    parser.add_argument("--harmonic-count", type=int, default=3)
    parser.add_argument("--confidence-floor", type=float, default=0.3)
    parser.add_argument("--dsp-device", default="cuda")
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-stem", default="drums")
    parser.add_argument("--listen-label", default="02_drums_rewritten.flac")
    parser.add_argument("--raw-ref-path")
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="export_hq_stem_rewrite_full.py",
        entrypoint_path="/app/tools/export_hq_stem_rewrite_full.py",
    )
    args = parse_args()
    source_stem_dir = Path(args.source_stem_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    listen_dir = Path(args.listen_dir).resolve() if args.listen_dir else None
    raw_ref_path = Path(args.raw_ref_path).resolve() if args.raw_ref_path else None

    if not source_stem_dir.exists():
        raise FileNotFoundError(f"Source stem dir is missing: {source_stem_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    manifest = export_aggressive_melody_first_from_stem_dir(
        source_stem_dir,
        output_dir,
        start_sec=None,
        end_sec=None,
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

    listen_summary: dict[str, str] | None = None
    if listen_dir is not None:
        selected_stem = str(args.listen_stem)
        stem_entry = manifest.get("stems", {}).get(selected_stem)
        if stem_entry is None:
            raise KeyError(f"Requested listen stem is missing from manifest: {selected_stem}")
        listen_dir.mkdir(parents=True, exist_ok=True)
        if raw_ref_path is not None:
            _write_flac_copy(raw_ref_path, listen_dir / "00_raw_ref.flac")
        _copy_listen_file(Path(str(stem_entry["path"])), listen_dir / args.listen_label)
        (listen_dir / "00_README.txt").write_text(
            "Listen only here.\n00 = raw reference.\n01 = base drums.\n02 = rewritten drums.\n",
            encoding="utf-8",
        )
        listen_summary = {
            "listen_dir": str(listen_dir),
            "listen_stem": selected_stem,
            "listen_label": str(args.listen_label),
        }

    summary = {
        "source_stem_dir": str(source_stem_dir),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest.get("manifest_path")),
        "elapsed_sec": elapsed_sec,
        "tuning": manifest.get("tuning"),
        "listen": listen_summary,
    }
    _emit(f"EXIT output_dir={output_dir} elapsed_sec={elapsed_sec}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
