from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

import soundfile as sf
import torchaudio


def _copy_listen_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def _save_preview_wav(
    source_audio_path: Path,
    output_wav_path: Path,
    *,
    start_sec: float,
    end_sec: float | None,
) -> tuple[int, int]:
    waveform, sample_rate = torchaudio.load(str(source_audio_path))
    start_frame = max(0, int(round(start_sec * sample_rate)))
    end_frame = waveform.shape[1] if end_sec is None else max(start_frame, int(round(end_sec * sample_rate)))
    preview = waveform[:, start_frame:end_frame]
    torchaudio.save(str(output_wav_path), preview, sample_rate)
    return int(sample_rate), int(preview.shape[1])


def _emit(message: str) -> None:
    print(f"[demucs_minus_preview] {message}", flush=True)


def main() -> None:
    require_docker_runtime(
        tool_name="export_demucs_minus_preview.py",
        entrypoint_path="/app/tools/export_demucs_minus_preview.py",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("source_audio_path")
    parser.add_argument("output_dir")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=40.0)
    parser.add_argument("--model-name", default="htdemucs_ft")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--segment", type=int, default=7)
    parser.add_argument("--shifts", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="01_demucs_minus.flac")
    args = parser.parse_args()

    source_audio_path = Path(args.source_audio_path)
    output_dir = Path(args.output_dir)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None

    if not source_audio_path.exists():
        raise FileNotFoundError(f"Source audio is missing: {source_audio_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    start_wall = time.time()

    with tempfile.TemporaryDirectory(prefix="demucs_minus_input_") as input_dir_str, tempfile.TemporaryDirectory(
        prefix="demucs_minus_output_"
    ) as demucs_output_dir_str:
        input_dir = Path(input_dir_str)
        demucs_output_dir = Path(demucs_output_dir_str)
        preview_input_path = input_dir / "preview.wav"

        _emit(
            f"ENTER source_audio={source_audio_path} output_dir={output_dir} start_sec={args.start_sec} end_sec={args.end_sec}"
        )
        sample_rate, frames = _save_preview_wav(
            source_audio_path,
            preview_input_path,
            start_sec=float(args.start_sec),
            end_sec=float(args.end_sec) if args.end_sec is not None else None,
        )
        _emit(f"prepared_preview_wav sample_rate={sample_rate} frames={frames}")

        command = [
            sys.executable,
            "-m",
            "demucs",
            "-n",
            str(args.model_name),
            "--two-stems=vocals",
            "--flac",
            "--int24",
            "--segment",
            str(args.segment),
            "--shifts",
            str(args.shifts),
            "--overlap",
            str(args.overlap),
            "-d",
            str(args.device),
            "-o",
            str(demucs_output_dir),
            str(preview_input_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        if completed.stdout.strip():
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
        if completed.stderr.strip():
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", flush=True)

        minus_source_path = demucs_output_dir / str(args.model_name) / "preview" / "no_vocals.flac"
        if not minus_source_path.exists():
            raise FileNotFoundError(f"Demucs minus output is missing: {minus_source_path}")

        minus_audio, minus_sr = sf.read(str(minus_source_path), always_2d=True)
        raw_audio, raw_sr = sf.read(str(preview_input_path), always_2d=True)

        minus_flac_path = output_dir / "minus.flac"
        raw_ref_flac_path = output_dir / "raw_ref.flac"
        sf.write(str(minus_flac_path), minus_audio, minus_sr, subtype="PCM_24")
        sf.write(str(raw_ref_flac_path), raw_audio, raw_sr, subtype="PCM_24")

        manifest = {
            "source_audio_path": str(source_audio_path),
            "preview_window_sec": {
                "start": float(args.start_sec),
                "end": None if args.end_sec is None else float(args.end_sec),
            },
            "model": {
                "name": "demucs_minus_preview_model",
                "model_name": str(args.model_name),
                "device": str(args.device),
                "segment": int(args.segment),
                "shifts": int(args.shifts),
                "overlap": float(args.overlap),
                "target_output": "no_vocals",
            },
            "artifacts": {
                "raw_ref_flac": str(raw_ref_flac_path),
                "minus_flac": str(minus_flac_path),
            },
            "audio": {
                "sample_rate": int(raw_sr),
                "frames": int(raw_audio.shape[0]),
                "duration_sec": round(float(raw_audio.shape[0]) / float(raw_sr), 6) if raw_sr else 0.0,
            },
            "elapsed_sec": round(time.time() - start_wall, 3),
        }
        manifest_path = output_dir / "stem_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        if listen_dir is not None:
            listen_dir.mkdir(parents=True, exist_ok=True)
            _copy_listen_file(raw_ref_flac_path, listen_dir / "00_raw_ref.flac")
            _copy_listen_file(minus_flac_path, listen_dir / args.listen_label)
            (listen_dir / "00_README.txt").write_text(
                "Listen only here.\n00 = raw reference.\n01 = Demucs no_vocals minus.\n",
                encoding="utf-8",
            )
            (listen_dir / "candidate_map.json").write_text(
                json.dumps(
                    {
                        "00_raw_ref.flac": str(raw_ref_flac_path),
                        args.listen_label: str(minus_flac_path),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        _emit(f"EXIT output_dir={output_dir} elapsed_sec={manifest['elapsed_sec']}")
        print(json.dumps({"output_dir": str(output_dir), "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
