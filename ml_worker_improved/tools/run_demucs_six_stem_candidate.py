"""Run Demucs htdemucs_6s and collect canonical six-stem outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.stem_contamination_common import STEMS


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _match_audio_shape(
    audio: np.ndarray,
    source_rate: int,
    *,
    target_sample_rate: int | None,
    target_sample_count: int | None,
) -> tuple[np.ndarray, int]:
    output = np.asarray(audio, dtype=np.float32)
    output_rate = int(source_rate)
    if target_sample_rate and int(target_sample_rate) != output_rate and len(output) > 0:
        old_x = np.linspace(0.0, 1.0, num=len(output), endpoint=False)
        new_len = max(1, int(round(len(output) * float(target_sample_rate) / float(output_rate))))
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        channels = [
            np.interp(new_x, old_x, output[:, channel]).astype(np.float32)
            for channel in range(output.shape[1])
        ]
        output = np.stack(channels, axis=1)
        output_rate = int(target_sample_rate)
    if target_sample_count is not None:
        desired = int(target_sample_count)
        if desired < 0:
            raise ValueError("target_sample_count must be non-negative")
        if len(output) > desired:
            output = output[:desired]
        elif len(output) < desired:
            pad = np.zeros((desired - len(output), output.shape[1]), dtype=np.float32)
            output = np.concatenate([output, pad], axis=0)
    return output.astype(np.float32), output_rate


def _copy_audio_as_flac(
    source_path: Path,
    output_path: Path,
    *,
    target_sample_rate: int | None = None,
    target_sample_count: int | None = None,
) -> Path:
    audio, sample_rate = sf.read(source_path, dtype="float32", always_2d=True)
    audio, sample_rate = _match_audio_shape(
        audio,
        int(sample_rate),
        target_sample_rate=target_sample_rate,
        target_sample_count=target_sample_count,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    return output_path


def collect_demucs_outputs(
    *,
    demucs_output_root: Path,
    model_name: str,
    track_name: str,
    stems_dir: Path,
    target_sample_rate: int | None = None,
    target_sample_count: int | None = None,
) -> dict[str, Any]:
    track_dir = demucs_output_root / model_name / track_name
    stem_paths: dict[str, str] = {}
    missing_stems: list[str] = []
    for stem in STEMS:
        source_path = track_dir / f"{stem}.wav"
        if not source_path.is_file():
            missing_stems.append(stem)
            continue
        stem_paths[stem] = str(
            _copy_audio_as_flac(
                source_path,
                stems_dir / f"{stem}.flac",
                target_sample_rate=target_sample_rate,
                target_sample_count=target_sample_count,
            )
        )
    return {
        "schema": "demucs_six_stem_collection.v1",
        "collection_status": "completed" if not missing_stems else "blocked_missing_stem",
        "demucs_track_dir": str(track_dir),
        "stem_paths": stem_paths,
        "missing_stems": missing_stems,
    }


def apply_residual_correction(*, input_audio: Path, stems_dir: Path, target_stem: str) -> dict[str, Any]:
    if target_stem not in STEMS:
        raise ValueError(f"unsupported residual correction stem: {target_stem}")
    mix_audio, mix_rate = sf.read(input_audio, dtype="float32", always_2d=True)
    summed = np.zeros_like(mix_audio, dtype=np.float32)
    missing: list[str] = []
    for stem in STEMS:
        stem_path = stems_dir / f"{stem}.flac"
        if not stem_path.is_file():
            missing.append(stem)
            continue
        stem_audio, stem_rate = sf.read(stem_path, dtype="float32", always_2d=True)
        if int(stem_rate) != int(mix_rate) or stem_audio.shape != mix_audio.shape:
            raise ValueError(f"stem shape mismatch for residual correction: {stem}")
        summed += stem_audio.astype(np.float32)
    if missing:
        return {"applied": False, "reason": "missing_stems", "missing_stems": missing}
    residual = mix_audio.astype(np.float32) - summed
    target_path = stems_dir / f"{target_stem}.flac"
    target_audio, _ = sf.read(target_path, dtype="float32", always_2d=True)
    sf.write(target_path, target_audio.astype(np.float32) + residual, mix_rate)
    return {
        "applied": True,
        "target_stem": target_stem,
        "residual_rms": float(np.sqrt(np.mean(np.square(residual.astype(np.float64))))),
    }


def run_demucs_six_stem_candidate(
    *,
    input_audio: Path,
    stems_dir: Path,
    work_dir: Path,
    model_name: str = "htdemucs_6s",
    segment_sec: float = 6.0,
    overlap: float = 0.25,
    shifts: int = 1,
    device: str = "cuda",
    residual_correction_stem: str = "",
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    demucs_output_root = work_dir / "demucs"
    input_info = sf.info(input_audio)
    segment_arg = str(int(segment_sec)) if float(segment_sec).is_integer() else str(segment_sec)
    command = [
        "demucs",
        "-n",
        model_name,
        "--segment",
        segment_arg,
        "--overlap",
        str(overlap),
        "--shifts",
        str(shifts),
        "-d",
        device,
        "-o",
        str(demucs_output_root),
        str(input_audio),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec, check=False)
    manifest: dict[str, Any] = {
        "schema": "demucs_six_stem_candidate.v1",
        "input_audio": str(input_audio),
        "stems_dir": str(stems_dir),
        "work_dir": str(work_dir),
        "model_name": model_name,
        "device": device,
        "residual_correction_stem": residual_correction_stem,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        manifest.update({"generation_status": "failed_runtime_error", "missing_stems": list(STEMS), "stem_paths": {}})
        return manifest
    collection = collect_demucs_outputs(
        demucs_output_root=demucs_output_root,
        model_name=model_name,
        track_name=input_audio.stem,
        stems_dir=stems_dir,
        target_sample_rate=int(input_info.samplerate),
        target_sample_count=int(input_info.frames),
    )
    manifest.update(collection)
    if not collection["missing_stems"] and residual_correction_stem:
        manifest["residual_correction"] = apply_residual_correction(
            input_audio=input_audio,
            stems_dir=stems_dir,
            target_stem=residual_correction_stem,
        )
    manifest["generation_status"] = "completed" if not collection["missing_stems"] else "blocked_missing_stem"
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-audio", required=True)
    parser.add_argument("--stems-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-name", default="htdemucs_6s")
    parser.add_argument("--segment-sec", type=float, default=6.0)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--shifts", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual-correction-stem", default="")
    parser.add_argument("--timeout-sec", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_demucs_six_stem_candidate(
        input_audio=Path(args.input_audio),
        stems_dir=Path(args.stems_dir),
        work_dir=Path(args.work_dir),
        model_name=args.model_name,
        segment_sec=args.segment_sec,
        overlap=args.overlap,
        shifts=args.shifts,
        device=args.device,
        residual_correction_stem=args.residual_correction_stem,
        timeout_sec=args.timeout_sec,
    )
    _write_json(Path(args.manifest), manifest)
    print(json.dumps(manifest, sort_keys=True))
    if manifest.get("generation_status") != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
