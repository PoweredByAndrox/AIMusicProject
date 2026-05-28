from __future__ import annotations

import argparse
import json
import os
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
from worker.high_quality_stem_export import (
    export_track_stems_aggressive_melody_first,
    export_track_stems_full_res,
)
from worker.stem_separator_defaults import (
    DEFAULT_STEM_EXPORT_INCLUDE_MELODY_STEM,
    DEFAULT_STEM_EXPORT_MODEL,
    DEFAULT_STEM_EXPORT_SHARPEN_BLEED,
)

DEFAULT_BS_ROFORMER_RUNTIME_REPO = Path("/tmp/msst_zfturbo")
DEFAULT_BS_ROFORMER_RUNTIME_WEIGHT_DIR = Path("/tmp/msst_zfturbo_weights")
DEFAULT_BS_ROFORMER_VENDOR_REPO = Path("/app/vendor/music_source_separation_training")
DEFAULT_BS_ROFORMER_VENDOR_CONFIG = Path("/app/vendor/bs_roformer_official/config_bs_roformer_384_8_2_485100.yaml")
DEFAULT_BS_ROFORMER_VENDOR_CKPT = Path("/app/vendor/bs_roformer_official/model_bs_roformer_ep_17_sdr_9.6568.ckpt")


def _emit(message: str) -> None:
    print(f"[hq_stems_full] {message}", flush=True)


def _copy_listen_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def _write_flac_copy(source_audio_path: Path, target_flac_path: Path) -> None:
    if source_audio_path.suffix.lower() == ".flac":
        shutil.copyfile(source_audio_path, target_flac_path)
        return
    audio, sample_rate = sf.read(str(source_audio_path), always_2d=True)
    sf.write(str(target_flac_path), audio, sample_rate, subtype="PCM_24")


def _ensure_link_or_copy(source_path: Path, target_path: Path) -> None:
    if target_path.exists():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source_path, target_path)
    except OSError:
        if source_path.is_dir():
            shutil.copytree(source_path, target_path)
        else:
            shutil.copyfile(source_path, target_path)


def _ensure_hybrid_bs_runtime() -> None:
    # The hybrid 6-stem path still expects the BS-Roformer repo and weights in /tmp.
    _ensure_link_or_copy(DEFAULT_BS_ROFORMER_VENDOR_REPO, DEFAULT_BS_ROFORMER_RUNTIME_REPO)
    DEFAULT_BS_ROFORMER_RUNTIME_WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_link_or_copy(
        DEFAULT_BS_ROFORMER_VENDOR_CONFIG,
        DEFAULT_BS_ROFORMER_RUNTIME_WEIGHT_DIR / "bs_roformer_vocals.yaml",
    )
    _ensure_link_or_copy(
        DEFAULT_BS_ROFORMER_VENDOR_CKPT,
        DEFAULT_BS_ROFORMER_RUNTIME_WEIGHT_DIR / "bs_roformer_vocals.ckpt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a full high-quality 6-stem separation from raw audio.")
    parser.add_argument("source_audio_path", help="Source full-track audio to process.")
    parser.add_argument("output_dir", help="Target directory for full stem artifacts.")
    parser.add_argument("--model-name", default=DEFAULT_STEM_EXPORT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sharpen-bleed", action="store_true", default=DEFAULT_STEM_EXPORT_SHARPEN_BLEED)
    parser.add_argument(
        "--aggressive-melody-first",
        action="store_true",
        help="Run the raw full-track aggressive melody-first export instead of the default full stem export.",
    )
    parser.add_argument(
        "--no-melody-stem",
        action="store_true",
        help="Disable the melody redistribution/export step.",
    )
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-stem", default="drums")
    parser.add_argument("--listen-label", default="01_drums.flac")
    parser.add_argument("--summary-path")
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="export_hq_stems_full.py",
        entrypoint_path="/app/tools/export_hq_stems_full.py",
    )
    args = parse_args()
    source_audio_path = Path(args.source_audio_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    listen_dir = Path(args.listen_dir).resolve() if args.listen_dir else None
    include_melody_stem = False if args.no_melody_stem else bool(DEFAULT_STEM_EXPORT_INCLUDE_MELODY_STEM)

    if not source_audio_path.exists():
        raise FileNotFoundError(f"Source audio is missing: {source_audio_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if str(args.model_name).startswith("hybrid_"):
        _ensure_hybrid_bs_runtime()
    started_at = time.perf_counter()
    if bool(args.aggressive_melody_first):
        manifest = export_track_stems_aggressive_melody_first(
            source_audio_path,
            output_dir,
            model_name=str(args.model_name),
            device=str(args.device),
            sharpen_bleed=bool(args.sharpen_bleed),
        )
        include_melody_stem = True
    else:
        manifest = export_track_stems_full_res(
            source_audio_path,
            output_dir,
            model_name=str(args.model_name),
            device=str(args.device),
            sharpen_bleed=bool(args.sharpen_bleed),
            include_melody_stem=bool(include_melody_stem),
        )
    elapsed_sec = round(time.perf_counter() - started_at, 3)

    listen_summary: dict[str, str] | None = None
    if listen_dir is not None:
        selected_stem = str(args.listen_stem)
        stem_entry = manifest.get("stems", {}).get(selected_stem)
        if stem_entry is None:
            raise KeyError(f"Requested listen stem is missing from manifest: {selected_stem}")
        listen_dir.mkdir(parents=True, exist_ok=True)
        raw_ref_path = listen_dir / "00_raw_ref.flac"
        selected_stem_path = Path(str(stem_entry["path"]))
        _write_flac_copy(source_audio_path, raw_ref_path)
        _copy_listen_file(selected_stem_path, listen_dir / args.listen_label)
        (listen_dir / "00_README.txt").write_text(
            "Listen only here.\n00 = raw reference.\n01 = selected isolated stem.\n",
            encoding="utf-8",
        )
        (listen_dir / "candidate_map.json").write_text(
            json.dumps(
                {
                    "00_raw_ref.flac": str(raw_ref_path),
                    args.listen_label: str(selected_stem_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        listen_summary = {
            "listen_dir": str(listen_dir),
            "listen_stem": selected_stem,
            "listen_label": str(args.listen_label),
        }

    summary = {
        "source_audio_path": str(source_audio_path),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest.get("manifest_path")),
        "model_name": str(args.model_name),
        "device": str(args.device),
        "bleed_sharpening": bool(args.sharpen_bleed),
        "aggressive_melody_first": bool(args.aggressive_melody_first),
        "melody_stem_included": bool(include_melody_stem),
        "elapsed_sec": elapsed_sec,
        "listen": listen_summary,
    }
    if args.summary_path:
        summary_path = Path(args.summary_path).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    _emit(f"EXIT output_dir={output_dir} elapsed_sec={elapsed_sec}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
