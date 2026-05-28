from __future__ import annotations

import contextlib
import gc
import io
import shutil
import struct
import traceback
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI

from app.schemas import RunRequest, RunResult
from app.utils import DRY_RUN

app = FastAPI(title="demucs-worker", version="0.2.0")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}


def output_root_for(request: RunRequest) -> Path:
    target = request.target_stem or request.task
    root = Path(request.output_root) / request.phase_id / f"{request.step_index:02d}_{target}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_demucs_args(request: RunRequest, output_root: Path) -> list[str]:
    model = request.model or "htdemucs_ft"
    # `--float32` keeps Demucs' WAV outputs in 32-bit IEEE float, aligning the
    # backbone with R-PCM (no silent 16-bit quantisation between pipeline
    # stages). Combined with the IEEE_FLOAT-correct _write_wave_file below.
    command = ["-n", model, "--float32", "--out", str(output_root)]
    if request.task == "coarse_6stem":
        # 6-stem path is intentionally pinned to htdemucs_6s because htdemucs_ft
        # has no guitar/piano heads. Per official Demucs README the 6s model has
        # weaker drums/bass/other than the fine-tuned 4-stem htdemucs_ft, so the
        # tandem presets should only consume guitar/piano from this task and use
        # coarse_4stem_ft (below) for the remaining stems.
        command = ["-n", "htdemucs_6s", "--float32", "--shifts", "2", "--overlap", "0.25", "--out", str(output_root)]
    elif request.task == "coarse_4stem_ft":
        # 4-stem fine-tuned path — best quality on vocals/drums/bass/other and
        # source for instrumental_mix in production tandem presets.
        command = ["-n", "htdemucs_ft", "--float32", "--shifts", "2", "--overlap", "0.25", "--out", str(output_root)]
    elif request.task == "coarse_vdbo_validation":
        command += ["--shifts", "2", "--overlap", "0.25"]
    elif request.task == "drums_focus":
        command += ["--two-stems", "drums", "--shifts", "2"]
    elif request.task == "bass_focus":
        command += ["--two-stems", "bass", "--shifts", "2"]
    command.append(request.input_path)
    return command


def build_command(request: RunRequest, output_root: Path) -> list[str]:
    return ["python", "-m", "demucs", *build_demucs_args(request, output_root)]


def planned_outputs(request: RunRequest, output_root: Path) -> dict[str, str]:
    if request.task == "coarse_6stem":
        stems = ["vocals", "drums", "bass", "guitar", "piano", "other"]
        named = {stem: str(output_root / f"{stem}.wav") for stem in stems}
        named["instrumental_mix"] = str(output_root / "instrumental_mix.wav")
        return named
    if request.task in {"coarse_4stem_ft", "coarse_vdbo_validation"}:
        stems = ["vocals", "drums", "bass", "other"]
        named = {stem: str(output_root / f"{stem}.wav") for stem in stems}
        named["instrumental_mix"] = str(output_root / "instrumental_mix.wav")
        return named
    if request.task == "drums_focus":
        return {"drums": str(output_root / "drums.wav")}
    if request.task == "bass_focus":
        return {"bass": str(output_root / "bass.wav")}
    target = request.target_stem or request.task
    return {target: str(output_root / f"{target}.wav")}


def _find_generated_file(output_root: Path, stem_key: str) -> Path | None:
    names = [stem_key]
    if stem_key == "instrumental_mix":
        names = ["instrumental_mix", "instrumental", "no_vocals", "minus"]

    candidates: list[Path] = []
    for file_path in output_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in AUDIO_EXTS:
            continue
        if file_path.stem.lower() in names:
            candidates.append(file_path)

    if not candidates:
        return None

    candidates.sort(key=lambda path: (len(path.relative_to(output_root).parts), len(str(path))))
    return candidates[0]


def _write_float_wav(path: Path, frames: np.ndarray, channels: int, samplerate: int) -> None:
    """Write a 32-bit IEEE_FLOAT WAV.

    The Python `wave` module always tags written WAVs as PCM (format=1), which
    means a 4-byte sample width with float bytes is misinterpreted by ffmpeg /
    soundfile as 32-bit signed PCM (audio comes out garbled). We emit the
    RIFF / fmt / fact / data chunks directly with the IEEE_FLOAT tag (format=3)
    so downstream tooling reads the file correctly. Cf. R-PCM in the
    stem-tandem-improvement-roadmap.
    """
    raw = frames.astype("<f4", copy=False).tobytes()
    bits_per_sample = 32
    bytes_per_sample = 4
    block_align = channels * bytes_per_sample
    byte_rate = int(samplerate) * block_align
    fmt_chunk_size = 16
    fact_chunk_size = 4
    num_samples = int(frames.shape[0])
    data_size = len(raw)
    riff_size = 4 + 8 + fmt_chunk_size + 8 + fact_chunk_size + 8 + data_size

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"RIFF")
        handle.write(struct.pack("<I", riff_size))
        handle.write(b"WAVE")
        handle.write(b"fmt ")
        handle.write(struct.pack("<I", fmt_chunk_size))
        handle.write(struct.pack("<H", 3))
        handle.write(struct.pack("<H", channels))
        handle.write(struct.pack("<I", int(samplerate)))
        handle.write(struct.pack("<I", byte_rate))
        handle.write(struct.pack("<H", block_align))
        handle.write(struct.pack("<H", bits_per_sample))
        handle.write(b"fact")
        handle.write(struct.pack("<I", fact_chunk_size))
        handle.write(struct.pack("<I", num_samples))
        handle.write(b"data")
        handle.write(struct.pack("<I", data_size))
        handle.write(raw)


def _write_wave_file(
    path: Path,
    audio: np.ndarray,
    samplerate: int,
    *,
    as_float: bool = True,
    bits_per_sample: int = 32,
) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
        audio = audio.T
    channels = int(audio.shape[0])
    frames = np.ascontiguousarray(audio.T)
    path.parent.mkdir(parents=True, exist_ok=True)

    if as_float or bits_per_sample == 32:
        _write_float_wav(path, frames, channels, samplerate)
        return

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setframerate(int(samplerate))

        if bits_per_sample == 24:
            clipped = np.clip(frames, -1.0, 1.0 - (1.0 / (1 << 23)))
            ints = np.round(clipped * ((1 << 23) - 1)).astype(np.int32, copy=False)
            packed = np.empty((ints.size, 3), dtype=np.uint8)
            packed[:, 0] = (ints & 0xFF).astype(np.uint8, copy=False)
            packed[:, 1] = ((ints >> 8) & 0xFF).astype(np.uint8, copy=False)
            packed[:, 2] = ((ints >> 16) & 0xFF).astype(np.uint8, copy=False)
            handle.setsampwidth(3)
            handle.writeframes(packed.tobytes())
            return

        clipped = np.clip(frames, -1.0, 1.0 - (1.0 / 32768.0))
        ints = np.round(clipped * 32767.0).astype("<i2", copy=False)
        handle.setsampwidth(2)
        handle.writeframes(ints.tobytes())


def _parse_wav_header(path: Path) -> tuple[int, int, int, int, bytes]:
    """Manual RIFF/WAVE parser that tolerates IEEE_FLOAT (format=3).

    Python's stdlib ``wave.open(rb)`` rejects any wFormatTag other than 1
    (``wave.Error: unknown format: 3``), so files written by ``_write_float_wav``
    (and by demucs's own float exporter) cannot be read back through
    ``wave``. We parse the chunks ourselves so the read path matches the
    write path symmetrically. Supports format codes 1 (PCM) and 3
    (IEEE_FLOAT); also unwraps WAVE_FORMAT_EXTENSIBLE (0xFFFE) to its
    inner SubFormat code. Returns ``(format_code, channels, samplerate,
    sample_width_bytes, raw_data_bytes)``.
    """
    raw_format_code: int | None = None
    channels: int | None = None
    samplerate: int | None = None
    sample_width: int | None = None
    data_bytes: bytes | None = None

    with path.open("rb") as handle:
        riff = handle.read(12)
        if len(riff) < 12 or riff[0:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"Not a RIFF/WAVE file: {path}")

        while True:
            header = handle.read(8)
            if len(header) < 8:
                break
            chunk_id = header[0:4]
            chunk_size = struct.unpack("<I", header[4:8])[0]
            chunk_data = handle.read(chunk_size)
            if chunk_size % 2 == 1:
                handle.read(1)
            if chunk_id == b"fmt ":
                if len(chunk_data) < 16:
                    raise ValueError(f"fmt chunk too short in {path}")
                raw_format_code = struct.unpack("<H", chunk_data[0:2])[0]
                channels = struct.unpack("<H", chunk_data[2:4])[0]
                samplerate = struct.unpack("<I", chunk_data[4:8])[0]
                bits_per_sample = struct.unpack("<H", chunk_data[14:16])[0]
                sample_width = bits_per_sample // 8
                if raw_format_code == 0xFFFE and len(chunk_data) >= 40:
                    sub_format = chunk_data[24:40]
                    inner_code = struct.unpack("<H", sub_format[0:2])[0]
                    raw_format_code = inner_code
            elif chunk_id == b"data":
                data_bytes = chunk_data

    if (
        raw_format_code is None
        or channels is None
        or samplerate is None
        or sample_width is None
        or data_bytes is None
    ):
        raise ValueError(f"Incomplete WAV header in {path}")

    return raw_format_code, int(channels), int(samplerate), int(sample_width), data_bytes


def _read_wave_file(path: Path) -> tuple[np.ndarray, int]:
    format_code, channels, samplerate, sample_width, raw = _parse_wav_header(path)

    if format_code == 3:
        if sample_width == 4:
            audio = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        elif sample_width == 8:
            audio = np.frombuffer(raw, dtype="<f8").astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported IEEE_FLOAT sample width: {sample_width} bytes ({path})"
            )
    elif format_code == 1:
        if sample_width == 2:
            audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
        elif sample_width == 3:
            triplets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            ints = (
                triplets[:, 0].astype(np.int32)
                | (triplets[:, 1].astype(np.int32) << 8)
                | (triplets[:, 2].astype(np.int32) << 16)
            )
            sign_bit = 1 << 23
            ints = np.where(ints & sign_bit, ints - (1 << 24), ints)
            audio = ints.astype(np.float32) / float((1 << 23) - 1)
        elif sample_width == 4:
            audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(
                (1 << 31) - 1
            )
        else:
            raise ValueError(
                f"Unsupported PCM sample width: {sample_width} bytes ({path})"
            )
    else:
        raise ValueError(
            f"Unsupported WAV format code {format_code} (only 1=PCM, 3=IEEE_FLOAT) for {path}"
        )

    audio = audio.reshape(-1, channels).T
    return audio, samplerate


def _custom_demucs_save_audio(
    wav_tensor,
    path: str | Path,
    samplerate: int,
    bitrate: int = 320,
    clip: str = "rescale",
    bits_per_sample: int = 32,
    as_float: bool = True,
    preset: int = 2,
) -> None:
    import demucs.audio

    del bitrate, preset

    destination = Path(path)
    if destination.suffix.lower() != ".wav":
        raise ValueError(f"Custom demucs worker only supports wav export, got: {destination.suffix}")

    clipped = demucs.audio.prevent_clip(wav_tensor, mode=clip)
    audio = clipped.detach().cpu().numpy()
    _write_wave_file(destination, audio, samplerate, as_float=as_float, bits_per_sample=bits_per_sample)


def _run_real_demucs(request: RunRequest, output_root: Path) -> tuple[int, str, str]:
    import demucs.audio
    import demucs.separate
    import torch

    demucs.audio.save_audio = _custom_demucs_save_audio
    demucs.separate.save_audio = _custom_demucs_save_audio

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    return_code = 0
    args = build_demucs_args(request, output_root)

    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        try:
            demucs.separate.main(args)
        except SystemExit as exc:
            code = exc.code
            if code is None:
                return_code = 0
            elif isinstance(code, int):
                return_code = code
            else:
                print(str(code), file=stderr_buffer)
                return_code = 1
        except Exception:
            traceback.print_exc(file=stderr_buffer)
            return_code = 1
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return return_code, stdout_buffer.getvalue(), stderr_buffer.getvalue()


def maybe_materialize_instrumental_mix(request: RunRequest, named_outputs: dict[str, str]) -> dict[str, str]:
    if request.task not in {"coarse_6stem", "coarse_4stem_ft", "coarse_vdbo_validation"}:
        return named_outputs
    if "instrumental_mix" in named_outputs:
        return named_outputs

    component_keys = [key for key in ("drums", "bass", "guitar", "piano", "other") if key in named_outputs]
    if not component_keys:
        component_keys = [key for key in named_outputs if key not in {"vocals", "instrumental_mix"}]
    if not component_keys:
        return named_outputs

    tracks: list[np.ndarray] = []
    samplerate: int | None = None
    min_frames: int | None = None
    for key in component_keys:
        stem_path = Path(named_outputs[key])
        if not stem_path.exists():
            continue
        audio, stem_samplerate = _read_wave_file(stem_path)
        if samplerate is None:
            samplerate = stem_samplerate
        elif samplerate != stem_samplerate:
            return named_outputs
        min_frames = audio.shape[1] if min_frames is None else min(min_frames, audio.shape[1])
        tracks.append(audio)

    if not tracks or samplerate is None or min_frames is None or min_frames <= 0:
        return named_outputs

    mixed = np.zeros((tracks[0].shape[0], min_frames), dtype=np.float32)
    for track in tracks:
        mixed += track[:, :min_frames]

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed /= peak

    destination = Path(planned_outputs(request, Path(request.output_root) / request.phase_id / f"{request.step_index:02d}_{request.target_stem or request.task}")["instrumental_mix"])
    _write_wave_file(destination, mixed, samplerate)
    named_outputs["instrumental_mix"] = str(destination)
    return named_outputs


def materialize_named_outputs(output_root: Path, named_outputs: dict[str, str]) -> dict[str, str]:
    materialized: dict[str, str] = {}
    for stem_key, destination in named_outputs.items():
        dest_path = Path(destination)
        if dest_path.exists():
            materialized[stem_key] = str(dest_path)
            continue

        source_path = _find_generated_file(output_root, stem_key)
        if source_path is None:
            continue

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != dest_path.resolve():
            shutil.copy2(source_path, dest_path)
        materialized[stem_key] = str(dest_path)
    return materialized


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "demucs-worker", "dry_run": DRY_RUN}


@app.get("/capabilities")
def capabilities() -> dict:
    return {
        "service": "demucs-worker",
        "tasks": [
            "coarse_4stem_ft",
            "coarse_6stem",
            "coarse_vdbo_validation",
            "drums_focus",
            "bass_focus",
        ],
        "models": ["htdemucs_ft", "htdemucs_6s", "htdemucs"],
        "state": "single-file ready; real run requires demucs package inside the container",
    }


@app.post("/run")
def run(request: RunRequest) -> dict:
    output_root = output_root_for(request)
    command = build_command(request, output_root)
    named_outputs = planned_outputs(request, output_root)
    metadata = {"input_mode": "single_file", "output_root": str(output_root)}

    status = "dry_run_completed"
    notes = "Dry-run command assembled. Set DRY_RUN=false and install demucs in the container to run real separation."
    if not request.dry_run:
        return_code, stdout, stderr = _run_real_demucs(request, output_root)
        metadata["stdout"] = stdout[-4000:]
        metadata["stderr"] = stderr[-4000:]
        status = "completed" if return_code == 0 else "failed"
        notes = "Real Demucs separation invoked with a custom wav exporter so stem files are written without TorchCodec."
        materialized_outputs = materialize_named_outputs(output_root, named_outputs)
        materialized_outputs = maybe_materialize_instrumental_mix(request, materialized_outputs)
        metadata["materialized_outputs"] = materialized_outputs
        if materialized_outputs:
            named_outputs = materialized_outputs

    result = RunResult(
        service="demucs-worker",
        status=status,
        task=request.task,
        target_stem=request.target_stem,
        command=command,
        output_files=list(named_outputs.values()),
        named_outputs=named_outputs,
        metadata=metadata,
        notes=notes,
    )
    return result.model_dump()
