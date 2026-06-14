from __future__ import annotations

import argparse
import json
import os
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


DEFAULT_MODEL_TYPE = "bs_roformer"
DEFAULT_TARGET_INSTRUMENT = "other"
DEFAULT_CONFIG_PATH = Path(
    "/app/vendor/bs_roformer_instrumental_resurrection_manual/config_bs_roformer_instrumental_resurrection_unwa.yaml"
)
DEFAULT_MODEL_PATH = Path(
    "/app/vendor/bs_roformer_instrumental_resurrection_manual/bs_roformer_instrumental_resurrection_unwa.ckpt"
)
DEFAULT_RUNTIME_REPO = Path("/app/vendor/music_source_separation_training")


def _emit(message: str) -> None:
    print(f"[instrumental_resurrection_minus] {message}", flush=True)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _mark_progress(
    event_log_path: Path,
    *,
    run_started_at: float,
    stage: str,
    status: str = "progress",
    stage_started_at: float | None = None,
    **metadata,
) -> float:
    now = time.perf_counter()
    event = {
        "type": "instrumental_resurrection_stage",
        "stage": str(stage),
        "status": str(status),
        "elapsed_sec": round(max(0.0, now - stage_started_at), 6) if stage_started_at is not None else 0.0,
        "since_start_sec": round(max(0.0, now - run_started_at), 6),
        "metadata": {
            str(key): _json_safe(value)
            for key, value in metadata.items()
            if value is not None
        },
    }
    with event_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)
    return now


def _resolve_bs_roformer_device(device: str) -> tuple[bool, list[str]]:
    requested = str(device).strip().lower()
    if requested in {"", "auto", "cuda"}:
        return False, ["0"]
    if requested in {"cpu", "cpu:0"}:
        return True, []
    normalized = requested.replace("cuda:", "").replace("cuda", "")
    normalized = normalized.replace(" ", ",")
    device_ids: list[str] = [
        token.strip()
        for token in normalized.split(",")
        if token.strip()
    ]
    if not device_ids:
        return False, ["0"]
    if any(not token.isdigit() for token in device_ids):
        raise ValueError(f"Unsupported BS Roformer device value: {device}")
    return False, device_ids


def _copy_listen_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def _resolve_output_by_stem(store_dir: Path, stem_name: str) -> Path:
    matches = sorted(
        path
        for path in store_dir.rglob(f"{stem_name}.*")
        if path.is_file()
    )
    if not matches:
        raise FileNotFoundError(f"BS Roformer {stem_name} output is missing in: {store_dir}")
    return matches[0]


def _run_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True, cwd=str(cwd))
    if completed.stdout.strip():
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.stderr.strip():
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", flush=True)
    completed.check_returncode()
    return completed


def _transcode_to_wav(source_audio_path: Path, target_wav_path: Path) -> tuple[int, int]:
    waveform, sample_rate = torchaudio.load(str(source_audio_path))
    torchaudio.save(
        str(target_wav_path),
        waveform,
        sample_rate,
        format="wav",
        encoding="PCM_S",
        bits_per_sample=24,
    )
    return int(sample_rate), int(waveform.shape[1])


def _write_flac_copy(source_audio_path: Path, target_flac_path: Path) -> tuple[int, int]:
    waveform, sample_rate = torchaudio.load(str(source_audio_path))
    torchaudio.save(
        str(target_flac_path),
        waveform,
        sample_rate,
        format="flac",
        bits_per_sample=24,
    )
    return int(sample_rate), int(waveform.shape[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a full-track Instrumental Resurrection minus from the accepted manual model recovery."
    )
    parser.add_argument("source_audio_path", help="Source full-track audio to process.")
    parser.add_argument("output_dir", help="Target directory for exported artifacts.")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--runtime-repo", default=str(DEFAULT_RUNTIME_REPO))
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--target-instrument", default=DEFAULT_TARGET_INSTRUMENT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="01_instrumental_resurrection_minus.flac")
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="export_instrumental_resurrection_minus.py",
        entrypoint_path="/app/tools/export_instrumental_resurrection_minus.py",
    )
    args = parse_args()
    source_audio_path = Path(args.source_audio_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    listen_dir = Path(args.listen_dir).resolve() if args.listen_dir else None
    runtime_repo = Path(args.runtime_repo).resolve()

    if not source_audio_path.exists():
        raise FileNotFoundError(f"Source audio is missing: {source_audio_path}")
    if not runtime_repo.exists():
        raise FileNotFoundError(f"BS Roformer runtime repo is missing: {runtime_repo}")
    if not (runtime_repo / "inference.py").exists():
        raise FileNotFoundError(f"BS Roformer runtime missing inference.py: {runtime_repo / 'inference.py'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    progress_log_path = output_dir / "instrumental_resurrection_progress.jsonl"
    progress_log_path.write_text("", encoding="utf-8")
    force_cpu, device_ids = _resolve_bs_roformer_device(args.device)
    execution_contract = {
        "route_name": "export_instrumental_resurrection_minus",
        "docker_runtime_required": True,
        "source_audio_path": str(source_audio_path),
        "output_dir": str(output_dir),
        "output_manifest_path": str(output_dir / "stem_manifest.json"),
        "progress_log_path": str(progress_log_path),
        "device": str(args.device),
        "target_instrument": str(args.target_instrument),
        "gpu_inference_skipped": bool(force_cpu),
        "gpu_inference_skip_reason": "cpu_requested" if force_cpu else None,
        "expected_acceleration_mode": "cpu_diagnostic_inference" if force_cpu else "gpu_inference",
    }
    _mark_progress(
        progress_log_path,
        run_started_at=started_at,
        stage="execution_contract",
        **execution_contract,
    )

    with tempfile.TemporaryDirectory(prefix="irs_input_") as input_dir_str, tempfile.TemporaryDirectory(
        prefix="irs_output_"
    ) as store_dir_str:
        input_dir = Path(input_dir_str)
        store_dir = Path(store_dir_str)
        input_wav_path = input_dir / f"{source_audio_path.stem}.wav"
        stage_started_at = _mark_progress(
            progress_log_path,
            run_started_at=started_at,
            stage="transcode_input",
            status="start",
            source_audio_path=str(source_audio_path),
            input_wav_path=str(input_wav_path),
        )
        sample_rate, frames = _transcode_to_wav(source_audio_path, input_wav_path)
        _mark_progress(
            progress_log_path,
            run_started_at=started_at,
            stage="transcode_input",
            status="finish",
            stage_started_at=stage_started_at,
            sample_rate=sample_rate,
            frames=frames,
            input_wav_path=str(input_wav_path),
        )
        _emit(f"prepared_input_wav sample_rate={sample_rate} frames={frames}")

        command = [
            sys.executable,
            "inference.py",
            "--model_type",
            str(args.model_type),
            "--config_path",
            str(args.config_path),
            "--start_check_point",
            str(args.model_path),
            "--input_folder",
            str(input_dir),
            "--store_dir",
            str(store_dir),
            "--filename_template",
            "{instr}",
            "--flac_file",
            "--pcm_type",
            "PCM_24",
            "--disable_detailed_pbar",
        ]
        if force_cpu:
            command.append("--force_cpu")
        else:
            command.extend(["--device_ids", *device_ids])
        if args.use_tta:
            command.append("--use_tta")
        stage_started_at = _mark_progress(
            progress_log_path,
            run_started_at=started_at,
            stage="model_inference",
            status="start",
            runtime_repo=str(runtime_repo),
            model_type=str(args.model_type),
            target_instrument=str(args.target_instrument),
            device=str(args.device),
            expected_acceleration_mode=execution_contract["expected_acceleration_mode"],
        )
        _run_command(command, cwd=runtime_repo)
        _mark_progress(
            progress_log_path,
            run_started_at=started_at,
            stage="model_inference",
            status="finish",
            stage_started_at=stage_started_at,
            store_dir=str(store_dir),
            target_instrument=str(args.target_instrument),
        )

        target_output_path = _resolve_output_by_stem(store_dir, args.target_instrument)
        if not target_output_path.exists():
            raise FileNotFoundError(f"Instrumental Resurrection target stem is missing: {target_output_path}")

        stage_started_at = _mark_progress(
            progress_log_path,
            run_started_at=started_at,
            stage="write_artifacts",
            status="start",
            target_output_path=str(target_output_path),
        )
        minus_audio, minus_sample_rate = sf.read(str(target_output_path), always_2d=True)
        raw_ref_flac_path = output_dir / "raw_ref.flac"
        minus_flac_path = output_dir / "instrumental_resurrection_minus.flac"

        _write_flac_copy(source_audio_path, raw_ref_flac_path)
        sf.write(str(minus_flac_path), minus_audio, minus_sample_rate, subtype="PCM_24")
        _mark_progress(
            progress_log_path,
            run_started_at=started_at,
            stage="write_artifacts",
            status="finish",
            stage_started_at=stage_started_at,
            raw_ref_flac=str(raw_ref_flac_path),
            instrumental_resurrection_minus_flac=str(minus_flac_path),
        )

        elapsed_sec = round(time.perf_counter() - started_at, 3)
        target_stem_artifact = {
            "target_stem": "minus",
            "path": str(minus_flac_path),
            "source": "instrumental_resurrection_model_output",
            "source_stems": [str(args.target_instrument)],
        }
        manifest = {
            "source_audio_path": str(source_audio_path),
            "target_stem": "minus",
            "target_audio_path": str(minus_flac_path),
            "target_stem_artifact": target_stem_artifact,
            "stems": {
                "minus": target_stem_artifact,
            },
            "instrumental_resurrection_output_path": str(minus_flac_path),
            "instrumental_resurrection_model_output_name": target_output_path.name,
            "artifacts": {
                "raw_ref_flac": str(raw_ref_flac_path),
                "instrumental_resurrection_minus_flac": str(minus_flac_path),
                "progress_log_jsonl": str(progress_log_path),
            },
            "model": {
                "name": "bs_roformer_instrumental_resurrection_unwa",
                "model_type": str(args.model_type),
                "config_path": str(args.config_path),
                "checkpoint_path": str(args.model_path),
                "target_instrument": str(args.target_instrument),
                "device": str(args.device),
                "registry_recovery": "manual_huggingface_mirror",
            },
            "audio": {
                "sample_rate": int(sample_rate),
                "frames": int(frames),
                "duration_sec": round(float(frames) / float(sample_rate), 6) if sample_rate else 0.0,
            },
            "execution_contract": execution_contract,
            "elapsed_sec": elapsed_sec,
        }
        manifest_path = output_dir / "stem_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        if listen_dir is not None:
            stage_started_at = _mark_progress(
                progress_log_path,
                run_started_at=started_at,
                stage="listen_pack",
                status="start",
                listen_dir=str(listen_dir),
            )
            listen_dir.mkdir(parents=True, exist_ok=True)
            _copy_listen_file(raw_ref_flac_path, listen_dir / "00_raw_ref.flac")
            _copy_listen_file(minus_flac_path, listen_dir / args.listen_label)
            (listen_dir / "00_README.txt").write_text(
                "Listen only here.\n00 = raw reference.\n01 = Instrumental Resurrection minus.\n",
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
            _mark_progress(
                progress_log_path,
                run_started_at=started_at,
                stage="listen_pack",
                status="finish",
                stage_started_at=stage_started_at,
                listen_dir=str(listen_dir),
            )

        _mark_progress(
            progress_log_path,
            run_started_at=started_at,
            stage="finish",
            status="finish",
            output_dir=str(output_dir),
            manifest_path=str(manifest_path),
            elapsed_sec=elapsed_sec,
        )
        _emit(f"EXIT output_dir={output_dir} elapsed_sec={elapsed_sec}")
        print(json.dumps({"output_dir": str(output_dir), "manifest_path": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
