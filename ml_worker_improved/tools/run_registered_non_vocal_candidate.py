from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from stem_contamination_common import normalize_stem_name  # noqa: E402


CommandRunner = Callable[[list[str], int], Any]
EXECUTION_MANIFEST_NAME = "registry_candidate_execution_manifest.json"
DEFAULT_MODEL_ID = "scnet_xl_ihf_musdb18"
REQUIRED_NON_VOCAL_STEMS = ("drums", "bass", "other")


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON manifest must be an object: {path}")
    return payload


def _default_command_runner(command: list[str], timeout_sec: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, timeout=timeout_sec, text=True)


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate)
    return path


def _audio_shape(path: Path) -> tuple[int, int, int]:
    audio, sample_rate = _read_audio(path)
    return int(sample_rate), int(audio.shape[0]), int(audio.shape[1])


def _match_audio_shape(
    audio: np.ndarray,
    source_rate: int,
    *,
    target_sample_rate: int,
    target_sample_count: int,
    target_channel_count: int,
) -> tuple[np.ndarray, int]:
    output = np.asarray(audio, dtype=np.float32)
    if output.ndim == 1:
        output = output[:, None]
    output_rate = int(source_rate)
    if int(target_sample_rate) != output_rate and len(output) > 0:
        old_x = np.linspace(0.0, 1.0, num=len(output), endpoint=False)
        new_len = max(1, int(round(len(output) * float(target_sample_rate) / float(output_rate))))
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        output = np.stack(
            [np.interp(new_x, old_x, output[:, channel]).astype(np.float32) for channel in range(output.shape[1])],
            axis=1,
        )
        output_rate = int(target_sample_rate)
    if len(output) > target_sample_count:
        output = output[:target_sample_count]
    elif len(output) < target_sample_count:
        pad = np.zeros((target_sample_count - len(output), output.shape[1]), dtype=np.float32)
        output = np.concatenate([output, pad], axis=0)
    if output.shape[1] != target_channel_count:
        if output.shape[1] == 1:
            output = np.repeat(output, target_channel_count, axis=1)
        elif target_channel_count == 1:
            output = np.mean(output, axis=1, keepdims=True).astype(np.float32)
        elif output.shape[1] > target_channel_count:
            output = output[:, :target_channel_count]
        else:
            pad = np.zeros((output.shape[0], target_channel_count - output.shape[1]), dtype=np.float32)
            output = np.concatenate([output, pad], axis=1)
    return output.astype(np.float32), output_rate


def _stem_audio_path(entry: Any, *, manifest_dir: Path) -> Path:
    if isinstance(entry, dict):
        raw_path = entry.get("path") or entry.get("audio_path") or entry.get("file")
    else:
        raw_path = entry
    if not raw_path:
        raise ValueError("stem entry does not contain an audio path")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = manifest_dir / path
    return path


def _normalized_stem_entries(stems: Mapping[str, Any]) -> dict[str, Any]:
    return {normalize_stem_name(stem): entry for stem, entry in stems.items()}


def _copy_required_stem(
    *,
    source_path: Path,
    target_path: Path,
    target_sample_rate: int,
    target_sample_count: int,
    target_channel_count: int,
) -> str:
    audio, sample_rate = _read_audio(source_path)
    audio, sample_rate = _match_audio_shape(
        audio,
        sample_rate,
        target_sample_rate=target_sample_rate,
        target_sample_count=target_sample_count,
        target_channel_count=target_channel_count,
    )
    return str(_write_audio(target_path, audio, sample_rate))


def _candidate_manifest_path(execution_manifest: Mapping[str, Any], registered_dir: Path) -> Path | None:
    candidate_manifest = execution_manifest.get("candidate_manifest")
    if isinstance(candidate_manifest, dict) and candidate_manifest.get("path"):
        path = Path(str(candidate_manifest["path"]))
        return path if path.is_absolute() else registered_dir / path
    fallback = registered_dir / "stem_manifest.json"
    return fallback if fallback.is_file() else None


def run_registered_non_vocal_candidate(
    *,
    input_audio: Path,
    stems_dir: Path,
    work_dir: Path,
    manifest_path: Path,
    command_runner: CommandRunner | None = None,
    timeout_sec: int = 900,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
) -> dict[str, Any]:
    runner = command_runner or _default_command_runner
    stems_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    target_sample_rate, target_sample_count, target_channel_count = _audio_shape(input_audio)
    registered_dir = work_dir / "registered"
    command = [
        "python",
        "tools/run_registered_model_candidate.py",
        str(input_audio),
        str(registered_dir),
        "--model-id",
        model_id,
        "--device",
        device,
    ]
    runner(command, timeout_sec)

    execution_manifest_path = registered_dir / EXECUTION_MANIFEST_NAME
    if not execution_manifest_path.is_file():
        manifest = {
            "schema": "registered_non_vocal_candidate.v1",
            "generation_status": "failed_missing_registered_manifest",
            "input_audio": str(input_audio),
            "stems_dir": str(stems_dir),
            "model_id": model_id,
            "registered_command": command,
            "registered_dir": str(registered_dir),
            "missing_stems": list(REQUIRED_NON_VOCAL_STEMS),
            "required_stems": list(REQUIRED_NON_VOCAL_STEMS),
        }
        _write_json(manifest_path, manifest)
        return manifest

    execution_manifest = _load_json(execution_manifest_path)
    candidate_manifest_path = _candidate_manifest_path(execution_manifest, registered_dir)
    if not candidate_manifest_path or not candidate_manifest_path.is_file():
        manifest = {
            "schema": "registered_non_vocal_candidate.v1",
            "generation_status": "failed_missing_candidate_manifest",
            "input_audio": str(input_audio),
            "stems_dir": str(stems_dir),
            "model_id": model_id,
            "registered_command": command,
            "registered_dir": str(registered_dir),
            "registered_manifest_path": str(execution_manifest_path),
            "missing_stems": list(REQUIRED_NON_VOCAL_STEMS),
            "required_stems": list(REQUIRED_NON_VOCAL_STEMS),
        }
        _write_json(manifest_path, manifest)
        return manifest

    candidate_manifest = _load_json(candidate_manifest_path)
    raw_stems = candidate_manifest.get("stems")
    stems = _normalized_stem_entries(raw_stems) if isinstance(raw_stems, dict) else {}
    missing_stems: list[str] = []
    stem_paths: dict[str, str] = {}
    manifest_dir = candidate_manifest_path.parent
    for stem in REQUIRED_NON_VOCAL_STEMS:
        entry = stems.get(stem)
        if entry is None:
            missing_stems.append(stem)
            continue
        try:
            source_path = _stem_audio_path(entry, manifest_dir=manifest_dir)
        except ValueError:
            missing_stems.append(stem)
            continue
        if not source_path.is_file():
            missing_stems.append(stem)
            continue
        stem_paths[stem] = _copy_required_stem(
            source_path=source_path,
            target_path=stems_dir / f"{stem}.flac",
            target_sample_rate=target_sample_rate,
            target_sample_count=target_sample_count,
            target_channel_count=target_channel_count,
        )

    manifest = {
        "schema": "registered_non_vocal_candidate.v1",
        "generation_status": "completed" if not missing_stems else "failed_missing_stem",
        "input_audio": str(input_audio),
        "stems_dir": str(stems_dir),
        "model_id": model_id,
        "device": device,
        "registered_command": command,
        "registered_dir": str(registered_dir),
        "registered_manifest_path": str(execution_manifest_path),
        "candidate_manifest_path": str(candidate_manifest_path),
        "stem_paths": stem_paths,
        "missing_stems": missing_stems,
        "required_stems": list(REQUIRED_NON_VOCAL_STEMS),
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one registered model candidate for drums/bass/other only.")
    parser.add_argument("--input-audio", required=True)
    parser.add_argument("--stems-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-sec", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    require_docker_runtime(
        tool_name="run_registered_non_vocal_candidate.py",
        entrypoint_path="/app/tools/run_registered_non_vocal_candidate.py",
    )
    args = parse_args()
    result = run_registered_non_vocal_candidate(
        input_audio=Path(args.input_audio),
        stems_dir=Path(args.stems_dir),
        work_dir=Path(args.work_dir),
        manifest_path=Path(args.manifest),
        model_id=str(args.model_id),
        device=str(args.device),
        timeout_sec=int(args.timeout_sec),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
