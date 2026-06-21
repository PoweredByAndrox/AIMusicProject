"""Build six stems by splitting a vocal separator's instrumental output."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import soundfile as sf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_fully_automatic_stem_search import _copy_audio_to_match_shape
from tools.stem_contamination_common import NON_VOCAL_STEMS, STEMS


CommandRunner = Callable[[list[str], int], None]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _audio_shape(path: Path) -> tuple[int, int, int]:
    info = sf.info(path)
    return int(info.samplerate), int(info.frames), int(info.channels)


def _default_command_runner(command: list[str], timeout_sec: int) -> None:
    subprocess.run(command, check=True, timeout=timeout_sec)


def _copy_required_stem(
    *,
    source_path: Path,
    target_path: Path,
    target_sample_rate: int,
    target_sample_count: int,
    target_channel_count: int,
) -> str:
    _copy_audio_to_match_shape(
        source_path,
        target_path,
        target_sample_rate=target_sample_rate,
        target_sample_count=target_sample_count,
        target_channel_count=target_channel_count,
    )
    return str(target_path)


def run_vocal_instrumental_demucs_candidate(
    *,
    input_audio: Path,
    stems_dir: Path,
    work_dir: Path,
    manifest_path: Path,
    command_runner: CommandRunner | None = None,
    timeout_sec: int = 900,
    vocal_model_filename: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    vocal_model_file_dir: str = "/app/models/audio_separator",
    vocal_model_id: str = "audio_separator_bs_roformer_vocals",
    vocal_model_display_name: str = "Audio Separator BS-RoFormer vocal checkpoint",
    instrumental_model_id: str = "",
    audio_separator_python: str = "/opt/audio-separator-venv/bin/python",
    demucs_model_name: str = "htdemucs_6s",
    demucs_segment_sec: str = "6",
    demucs_overlap: str = "0.25",
    demucs_shifts: str = "1",
    device: str = "cuda",
) -> dict[str, Any]:
    runner = command_runner or _default_command_runner
    stems_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    target_sample_rate, target_sample_count, target_channel_count = _audio_shape(input_audio)

    pair_stems_dir = work_dir / "pair" / "stems"
    pair_command = [
        "python",
        "tools/export_audio_separator_vocal_full.py",
        str(input_audio),
        str(pair_stems_dir),
        "--model-filename",
        vocal_model_filename,
        "--model-file-dir",
        vocal_model_file_dir,
        "--audio-separator-python",
        audio_separator_python,
        "--manifest-model-id",
        vocal_model_id,
        "--manifest-model-display-name",
        vocal_model_display_name,
        "--require-cuda-provider",
    ]
    runner(pair_command, timeout_sec)

    vocal_path = pair_stems_dir / "vocals.flac"
    pair_instrumental_path = pair_stems_dir / "instrumental.flac"
    missing_pair_outputs = [
        str(path.name)
        for path in (vocal_path, pair_instrumental_path)
        if not path.is_file()
    ]
    if missing_pair_outputs:
        manifest = {
            "schema": "vocal_instrumental_demucs_candidate.v1",
            "generation_status": "failed_missing_vocal_pair_output",
            "missing_outputs": missing_pair_outputs,
            "input_audio": str(input_audio),
            "stems_dir": str(stems_dir),
        }
        _write_json(manifest_path, manifest)
        return manifest

    instrumental_path = pair_instrumental_path
    instrumental_command: list[str] = []
    instrumental_manifest_path: Path | None = None
    if instrumental_model_id:
        instrumental_dir = work_dir / "instrumental_model"
        instrumental_command = [
            "python",
            "tools/run_registered_model_candidate.py",
            str(input_audio),
            str(instrumental_dir),
            "--model-id",
            instrumental_model_id,
            "--device",
            device,
            "--target-stem",
            "minus",
        ]
        runner(instrumental_command, timeout_sec)
        instrumental_manifest_path = instrumental_dir / "registry_candidate_execution_manifest.json"
        if not instrumental_manifest_path.is_file():
            manifest = {
                "schema": "vocal_instrumental_demucs_candidate.v1",
                "generation_status": "failed_missing_instrumental_model_manifest",
                "input_audio": str(input_audio),
                "stems_dir": str(stems_dir),
                "instrumental_model_id": instrumental_model_id,
                "instrumental_command": instrumental_command,
            }
            _write_json(manifest_path, manifest)
            return manifest
        instrumental_manifest = json.loads(instrumental_manifest_path.read_text(encoding="utf-8-sig"))
        target_audio_path = (
            instrumental_manifest.get("candidate_manifest", {}).get("target_audio_path")
            if isinstance(instrumental_manifest.get("candidate_manifest"), dict)
            else None
        )
        if not target_audio_path or not Path(str(target_audio_path)).is_file():
            manifest = {
                "schema": "vocal_instrumental_demucs_candidate.v1",
                "generation_status": "failed_missing_instrumental_model_audio",
                "input_audio": str(input_audio),
                "stems_dir": str(stems_dir),
                "instrumental_model_id": instrumental_model_id,
                "instrumental_command": instrumental_command,
                "instrumental_manifest_path": str(instrumental_manifest_path),
            }
            _write_json(manifest_path, manifest)
            return manifest
        instrumental_path = Path(str(target_audio_path))

    demucs_stems_dir = work_dir / "demucs_instrumental" / "stems"
    demucs_manifest_path = work_dir / "demucs_instrumental" / "demucs.json"
    demucs_command = [
        "python",
        "tools/run_demucs_six_stem_candidate.py",
        "--input-audio",
        str(instrumental_path),
        "--stems-dir",
        str(demucs_stems_dir),
        "--work-dir",
        str(work_dir / "demucs_instrumental" / "work"),
        "--manifest",
        str(demucs_manifest_path),
        "--model-name",
        demucs_model_name,
        "--segment-sec",
        str(demucs_segment_sec),
        "--overlap",
        str(demucs_overlap),
        "--shifts",
        str(demucs_shifts),
        "--device",
        device,
        "--timeout-sec",
        str(timeout_sec),
    ]
    runner(demucs_command, timeout_sec)

    stem_paths: dict[str, str] = {}
    missing_stems: list[str] = []
    vocal_source = vocal_path
    if not vocal_source.is_file():
        missing_stems.append("vocals")
    else:
        stem_paths["vocals"] = _copy_required_stem(
            source_path=vocal_source,
            target_path=stems_dir / "vocals.flac",
            target_sample_rate=target_sample_rate,
            target_sample_count=target_sample_count,
            target_channel_count=target_channel_count,
        )
    for stem in NON_VOCAL_STEMS:
        source = demucs_stems_dir / f"{stem}.flac"
        if not source.is_file():
            missing_stems.append(stem)
            continue
        stem_paths[stem] = _copy_required_stem(
            source_path=source,
            target_path=stems_dir / f"{stem}.flac",
            target_sample_rate=target_sample_rate,
            target_sample_count=target_sample_count,
            target_channel_count=target_channel_count,
        )

    manifest = {
        "schema": "vocal_instrumental_demucs_candidate.v1",
        "generation_status": "completed" if not missing_stems else "failed_missing_stem",
        "input_audio": str(input_audio),
        "instrumental_path": str(instrumental_path),
        "pair_instrumental_path": str(pair_instrumental_path),
        "instrumental_model_id": instrumental_model_id,
        "instrumental_command": instrumental_command,
        "instrumental_manifest_path": str(instrumental_manifest_path) if instrumental_manifest_path else "",
        "vocal_model_filename": vocal_model_filename,
        "vocal_model_file_dir": vocal_model_file_dir,
        "vocal_model_id": vocal_model_id,
        "vocal_model_display_name": vocal_model_display_name,
        "vocal_pair_command": pair_command,
        "demucs_input_audio": str(instrumental_path),
        "demucs_command": demucs_command,
        "stems_dir": str(stems_dir),
        "stem_paths": stem_paths,
        "missing_stems": missing_stems,
        "required_stems": list(STEMS),
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-audio", required=True)
    parser.add_argument("--stems-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vocal-model-filename", default="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    parser.add_argument("--vocal-model-file-dir", default="/app/models/audio_separator")
    parser.add_argument("--vocal-model-id", default="audio_separator_bs_roformer_vocals")
    parser.add_argument("--vocal-model-display-name", default="Audio Separator BS-RoFormer vocal checkpoint")
    parser.add_argument("--instrumental-model-id", default="")
    parser.add_argument("--audio-separator-python", default="/opt/audio-separator-venv/bin/python")
    parser.add_argument("--demucs-model-name", default="htdemucs_6s")
    parser.add_argument("--segment-sec", default="6")
    parser.add_argument("--overlap", default="0.25")
    parser.add_argument("--shifts", default="1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-sec", type=int, default=900)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_vocal_instrumental_demucs_candidate(
        input_audio=Path(args.input_audio),
        stems_dir=Path(args.stems_dir),
        work_dir=Path(args.work_dir),
        manifest_path=Path(args.manifest),
        timeout_sec=int(args.timeout_sec),
        vocal_model_filename=str(args.vocal_model_filename),
        vocal_model_file_dir=str(args.vocal_model_file_dir),
        vocal_model_id=str(args.vocal_model_id),
        vocal_model_display_name=str(args.vocal_model_display_name),
        instrumental_model_id=str(args.instrumental_model_id),
        audio_separator_python=str(args.audio_separator_python),
        demucs_model_name=str(args.demucs_model_name),
        demucs_segment_sec=str(args.segment_sec),
        demucs_overlap=str(args.overlap),
        demucs_shifts=str(args.shifts),
        device=str(args.device),
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0 if manifest["generation_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
