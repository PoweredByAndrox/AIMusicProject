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
import yaml


DEFAULT_REPO_PATH = Path("/app/vendor/music_source_separation_training")
DEFAULT_MODEL_DIR = Path("/app/vendor/dereverb_echo_melband")
DEFAULT_CONFIG_PATH = DEFAULT_MODEL_DIR / "config_dereverb_echo_mbr_v2.yaml"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt"
DEFAULT_MODEL_TYPE = "mel_band_roformer"


def _emit(message: str) -> None:
    print(f"[dereverb_vocal_preview] {message}", flush=True)


def _copy_listen_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def validate_preview_window(start_sec: float, end_sec: float | None) -> None:
    if float(start_sec) < 0.0:
        raise ValueError(f"start_sec must be >= 0.0, got {start_sec}")
    if end_sec is not None and float(end_sec) <= float(start_sec):
        raise ValueError(f"end_sec must be greater than start_sec, got {start_sec}:{end_sec}")


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


def _write_config_with_overlap(
    config_path: Path,
    target_path: Path,
    *,
    num_overlap: int,
) -> None:
    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    config.setdefault("inference", {})
    config["inference"]["num_overlap"] = int(num_overlap)
    target_path.write_text(yaml.dump(config, sort_keys=False), encoding="utf-8")


copy_listen_file = _copy_listen_file
save_audio_window_as_wav = _save_preview_wav
write_config_with_overlap = _write_config_with_overlap


def main() -> None:
    require_docker_runtime(
        tool_name="export_dereverb_vocal_preview.py",
        entrypoint_path="/app/tools/export_dereverb_vocal_preview.py",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("source_audio_path")
    parser.add_argument("output_dir")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=40.0)
    parser.add_argument("--repo-path", default=str(DEFAULT_REPO_PATH))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-overlap", type=int, default=2)
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="01_dereverb_echo_melband_dry.flac")
    parser.add_argument("--target-output", default="dry")
    parser.add_argument("--output-file-name", default="vocals.flac")
    parser.add_argument("--extract-instrumental", action="store_true")
    args = parser.parse_args()

    source_audio_path = Path(args.source_audio_path)
    output_dir = Path(args.output_dir)
    repo_path = Path(args.repo_path)
    config_path = Path(args.config_path)
    model_path = Path(args.model_path)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None

    if not source_audio_path.exists():
        raise FileNotFoundError(f"Source audio is missing: {source_audio_path}")
    if not repo_path.exists():
        raise FileNotFoundError(f"MSST repo is missing: {repo_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Dereverb config is missing: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Dereverb model is missing: {model_path}")
    validate_preview_window(float(args.start_sec), float(args.end_sec) if args.end_sec is not None else None)

    output_dir.mkdir(parents=True, exist_ok=True)
    start_wall = time.time()

    with tempfile.TemporaryDirectory(prefix="dereverb_preview_input_") as input_dir_str, tempfile.TemporaryDirectory(
        prefix="dereverb_preview_output_"
    ) as model_output_dir_str, tempfile.TemporaryDirectory(prefix="dereverb_preview_config_") as config_dir_str:
        input_dir = Path(input_dir_str)
        model_output_dir = Path(model_output_dir_str)
        config_dir = Path(config_dir_str)
        preview_input_path = input_dir / "preview.wav"
        runtime_config_path = config_dir / "config_runtime.yaml"

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
        _write_config_with_overlap(config_path, runtime_config_path, num_overlap=int(args.num_overlap))

        command = [
            sys.executable,
            "inference.py",
            "--model_type",
            str(args.model_type),
            "--config_path",
            str(runtime_config_path),
            "--start_check_point",
            str(model_path),
            "--input_folder",
            str(input_dir),
            "--store_dir",
            str(model_output_dir),
            "--device_ids",
            str(args.device_id),
            "--disable_detailed_pbar",
            "--filename_template",
            "{file_name}/{instr}",
        ]
        if args.extract_instrumental:
            command.append("--extract_instrumental")
        completed = subprocess.run(
            command,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.stdout.strip():
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
        if completed.stderr.strip():
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", flush=True)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
                output=completed.stdout,
                stderr=completed.stderr,
            )

        dry_wav = model_output_dir / "preview" / f"{args.target_output}.wav"
        if not dry_wav.exists():
            raise FileNotFoundError(f"Dereverb dry output is missing: {dry_wav}")

        dry_audio, dry_sr = sf.read(str(dry_wav), always_2d=True)
        raw_audio, raw_sr = sf.read(str(preview_input_path), always_2d=True)

        output_flac_path = output_dir / args.output_file_name
        raw_ref_flac_path = output_dir / "raw_ref.flac"
        sf.write(str(output_flac_path), dry_audio, dry_sr, subtype="PCM_24")
        sf.write(str(raw_ref_flac_path), raw_audio, raw_sr, subtype="PCM_24")

        manifest = {
            "source_audio_path": str(source_audio_path),
            "preview_window_sec": {
                "start": float(args.start_sec),
                "end": None if args.end_sec is None else float(args.end_sec),
            },
            "model": {
                "name": f"{args.model_type}_preview_model",
                "repo_path": str(repo_path),
                "config_path": str(runtime_config_path),
                "config_template_path": str(config_path),
                "model_path": str(model_path),
                "device_id": int(args.device_id),
                "num_overlap": int(args.num_overlap),
                "target_output": str(args.target_output),
                "model_type": str(args.model_type),
            },
            "artifacts": {
                "raw_ref_flac": str(raw_ref_flac_path),
                "output_flac": str(output_flac_path),
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
            _copy_listen_file(output_flac_path, listen_dir / args.listen_label)
            (listen_dir / "00_README.txt").write_text(
                "Listen only here.\n00 = raw baseline preview.\n01 = processed target-output preview.\n",
                encoding="utf-8",
            )
            (listen_dir / "candidate_map.json").write_text(
                json.dumps(
                    {
                        "00_raw_ref.flac": str(raw_ref_flac_path),
                        args.listen_label: str(output_flac_path),
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
