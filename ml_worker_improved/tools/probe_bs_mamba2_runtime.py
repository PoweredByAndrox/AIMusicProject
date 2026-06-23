from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from export_bs_mamba2_vocal_preview import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPO_PATH,
)


MODEL_ID = "bs_mamba2_vocals"
MARKER_NAME = "runtime_probe_passed.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_synthetic_probe_audio(path: Path, *, duration_sec: float = 2.0, sample_rate: int = 48000) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(round(float(duration_sec) * int(sample_rate))))
    t = np.arange(frame_count, dtype=np.float32) / float(sample_rate)
    left = 0.035 * np.sin(2.0 * math.pi * 220.0 * t)
    right = 0.030 * np.sin(2.0 * math.pi * 330.0 * t + 0.25)
    audio = np.stack([left, right], axis=1).astype(np.float32)
    sf.write(str(path), audio, int(sample_rate), subtype="PCM_24")
    return {
        "path": str(path),
        "sample_rate": int(sample_rate),
        "frames": int(frame_count),
        "duration_sec": round(float(frame_count) / float(sample_rate), 6),
        "sha256": _sha256_file(path),
    }


def _manifest_stem_path(entry: Any, *, manifest_dir: Path) -> Path | None:
    if isinstance(entry, dict):
        raw_path = str(entry.get("path") or "").strip()
    else:
        raw_path = str(entry or "").strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_dir / path
    return path


def validate_probe_outputs(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "stem_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("missing_probe_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stems = manifest.get("stems") if isinstance(manifest.get("stems"), dict) else {}
    output_info: dict[str, Any] = {}
    for stem in ("vocals", "instrumental"):
        stem_path = _manifest_stem_path(stems.get(stem), manifest_dir=manifest_path.parent)
        if stem_path is None or not stem_path.exists():
            raise RuntimeError(f"missing_probe_output:{stem}")
        info = sf.info(str(stem_path))
        if int(info.frames) <= 0 or int(info.samplerate) <= 0 or int(info.channels) <= 0:
            raise RuntimeError(f"invalid_probe_output:{stem}")
        output_info[stem] = {
            "path": str(stem_path),
            "sample_rate": int(info.samplerate),
            "frames": int(info.frames),
            "channels": int(info.channels),
            "duration_sec": round(float(info.frames) / float(info.samplerate), 6),
            "sha256": _sha256_file(stem_path),
        }
    return {
        "manifest_path": str(manifest_path),
        "manifest_model_id": str(manifest.get("model_id") or manifest.get("model_name") or ""),
        "outputs": output_info,
    }


def build_success_marker_payload(
    *,
    source_audio_path: Path,
    output_dir: Path,
    repo_path: Path,
    model_path: Path,
    config_path: Path,
    output_validation: dict[str, Any],
    elapsed_sec: float,
) -> dict[str, Any]:
    return {
        "schema": "bs_mamba2_runtime_probe.v1",
        "model_id": MODEL_ID,
        "status": "passed",
        "runtime_probe_status": "passed",
        "created_at": _utc_now(),
        "contract": {
            "docker_only": True,
            "synthetic_probe_audio": True,
            "fallback_used": False,
            "hidden_inference": False,
            "manual_marker": False,
            "delivery_artifact": False,
        },
        "source_audio_path": str(source_audio_path),
        "output_dir": str(output_dir),
        "repo_path": str(repo_path),
        "model_path": str(model_path),
        "config_path": str(config_path),
        "validated_outputs": ["vocals", "instrumental"],
        "output_validation": output_validation,
        "elapsed_sec": round(float(elapsed_sec), 3),
    }


def write_success_marker_from_probe_outputs(
    *,
    marker_path: Path,
    output_dir: Path,
    source_audio_path: Path,
    repo_path: Path,
    model_path: Path,
    config_path: Path,
    elapsed_sec: float,
) -> dict[str, Any]:
    if not source_audio_path.exists():
        raise RuntimeError("missing_probe_source_audio")
    output_validation = validate_probe_outputs(output_dir)
    payload = build_success_marker_payload(
        source_audio_path=source_audio_path,
        output_dir=output_dir,
        repo_path=repo_path,
        model_path=model_path,
        config_path=config_path,
        output_validation=output_validation,
        elapsed_sec=elapsed_sec,
    )
    write_marker_payload(marker_path, payload)
    return payload


def write_marker_payload(marker_path: Path, payload: dict[str, Any]) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = marker_path.with_suffix(marker_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(marker_path)


def prepare_probe_output_dir(output_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("probe_output_dir_must_be_empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_probe(
    *,
    marker_path: Path,
    repo_path: Path,
    model_path: Path,
    config_path: Path,
    duration_sec: float,
    sample_rate: int,
    device_id: int,
    num_overlap: int,
    keep_output_dir: Path | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="bs_mamba2_probe_") as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        source_audio_path = tmp_dir / "probe.flac"
        output_dir = prepare_probe_output_dir(keep_output_dir or (tmp_dir / "out"))
        source_summary = write_synthetic_probe_audio(
            source_audio_path,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
        )
        command = [
            sys.executable,
            str(TOOLS_DIR / "export_bs_mamba2_vocal_full.py"),
            str(source_audio_path),
            str(output_dir),
            "--repo-path",
            str(repo_path),
            "--config-path",
            str(config_path),
            "--model-path",
            str(model_path),
            "--device-id",
            str(device_id),
            "--num-overlap",
            str(num_overlap),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)
        marker = write_success_marker_from_probe_outputs(
            marker_path=marker_path,
            output_dir=output_dir,
            source_audio_path=source_audio_path,
            repo_path=repo_path,
            model_path=model_path,
            config_path=config_path,
            elapsed_sec=time.monotonic() - start,
        )
        marker["source_audio"] = source_summary
        write_marker_payload(marker_path, marker)
        return marker


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded BS-Mamba2 runtime probe and write success marker.")
    parser.add_argument("--repo-path", type=Path, default=DEFAULT_REPO_PATH)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--marker-path", type=Path, default=DEFAULT_MODEL_DIR / MARKER_NAME)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-overlap", type=int, default=1)
    parser.add_argument("--keep-output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="probe_bs_mamba2_runtime.py",
        entrypoint_path="/app/tools/probe_bs_mamba2_runtime.py",
    )
    args = build_arg_parser().parse_args(argv)
    for label, path in (
        ("repo", args.repo_path),
        ("config", args.config_path),
        ("model", args.model_path),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"BS-Mamba2 {label} path is missing: {path}")
    marker = run_probe(
        marker_path=args.marker_path,
        repo_path=args.repo_path,
        model_path=args.model_path,
        config_path=args.config_path,
        duration_sec=float(args.duration_sec),
        sample_rate=int(args.sample_rate),
        device_id=int(args.device_id),
        num_overlap=int(args.num_overlap),
        keep_output_dir=args.keep_output_dir,
    )
    print(json.dumps({"marker_path": str(args.marker_path), "status": marker["runtime_probe_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
