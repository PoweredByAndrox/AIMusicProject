from __future__ import annotations

import json
import math
import shutil
import gc
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI

from app.schemas import RunRequest, RunResult
from app.utils import (
    DRY_RUN,
    LYRICS_ALIGN_MODEL,
    LYRICS_BATCH_SIZE,
    LYRICS_WHISPERX_MODEL,
    choose_compute_type,
    choose_device,
)

app = FastAPI(title="lyrics-worker", version="0.2.0")

LYRICS_SUBPROCESS_SCRIPT = "/app/scripts/run_lyrics_task.py"


def release_inference_memory(*objects: object) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_lyrics_subprocess(mode: str, payload: dict[str, Any]) -> tuple[dict, dict, list[str]]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"/app{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "/app"
    completed = subprocess.run(
        ["python", LYRICS_SUBPROCESS_SCRIPT, mode],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"lyrics subprocess failed with code {completed.returncode}"
        raise RuntimeError(detail)

    stdout_text = completed.stdout.strip()
    json_line = stdout_text.splitlines()[-1] if stdout_text else ""
    decoded = json.loads(json_line)
    return decoded["payload"], decoded["metadata"], decoded["command_preview"]


def output_root_for(request: RunRequest) -> Path:
    target = request.target_stem or request.task
    root = Path(request.output_root) / request.phase_id / f"{request.step_index:02d}_{target}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _format_lrc_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"[{minutes:02d}:{remainder:05.2f}]"


def _format_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - math.floor(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _coerce_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_segments(raw_segments: List[dict]) -> List[dict]:
    normalized: List[dict] = []
    for segment in raw_segments or []:
        start = segment.get("start")
        end = segment.get("end")
        if start is None or end is None:
            continue
        words: List[dict] = []
        for word in segment.get("words", []) or []:
            word_start = word.get("start")
            word_end = word.get("end")
            word_text = _coerce_text(word.get("word") or word.get("text"))
            if word_start is None or word_end is None or not word_text:
                continue
            item = {
                "start": float(word_start),
                "end": float(word_end),
                "text": word_text,
            }
            confidence = word.get("confidence")
            if confidence is not None:
                item["confidence"] = confidence
            words.append(item)
        text = _coerce_text(segment.get("text"))
        if not text and words:
            text = " ".join(item["text"] for item in words)
        normalized.append(
            {
                "start": float(start),
                "end": float(end),
                "text": text,
                "words": words,
            }
        )
    return normalized


def payload_from_segments(source: str, language: Optional[str], raw_segments: List[dict], raw_text: Optional[str] = None, extra: Optional[dict] = None) -> dict:
    segments = normalize_segments(raw_segments)
    full_text = _coerce_text(raw_text) or "\n".join(seg["text"] for seg in segments if seg["text"])
    payload = {
        "source": source,
        "language": language,
        "text": full_text,
        "segments": segments,
        "segment_count": len(segments),
        "word_count": sum(len(seg.get("words", [])) for seg in segments),
    }
    if extra:
        payload["extra"] = extra
    return payload


def write_payload_files(prefix: str, payload: dict, output_root: Path) -> dict[str, str]:
    base = output_root / prefix
    json_path = base.with_suffix(".json")
    txt_path = base.with_suffix(".txt")
    lrc_path = base.with_suffix(".lrc")
    srt_path = base.with_suffix(".srt")

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(payload.get("text", "").strip() + "\n", encoding="utf-8")

    lrc_lines = []
    srt_blocks = []
    for idx, seg in enumerate(payload.get("segments", []), start=1):
        text = _coerce_text(seg.get("text"))
        start = float(seg["start"])
        end = float(seg["end"])
        lrc_lines.append(f"{_format_lrc_time(start)}{text}")
        srt_blocks.append(f"{idx}\n{_format_srt_time(start)} --> {_format_srt_time(end)}\n{text}\n")
    lrc_path.write_text("\n".join(lrc_lines) + ("\n" if lrc_lines else ""), encoding="utf-8")
    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")

    named = {
        f"{prefix}_json": str(json_path),
        f"{prefix}_txt": str(txt_path),
        f"{prefix}_lrc": str(lrc_path),
        f"{prefix}_srt": str(srt_path),
        "lyrics_json": str(json_path),
        "lyrics_txt": str(txt_path),
        "lyrics_lrc": str(lrc_path),
        "lyrics_srt": str(srt_path),
    }
    return named


def infer_language(request: RunRequest) -> Optional[str]:
    if request.lyrics_language:
        return request.lyrics_language
    for key, path_str in request.build_from_paths.items():
        if not key.endswith("_json"):
            continue
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        language = data.get("language")
        if language:
            return language
    return None


def transcribe_with_whisperx(audio_path: Path, language: Optional[str]) -> tuple[dict, dict, list[str]]:
    import whisperx

    device = choose_device()
    compute_type = choose_compute_type(device)
    command_preview = [
        "whisperx",
        str(audio_path),
        "--model",
        LYRICS_WHISPERX_MODEL,
        "--output_format",
        "json",
        "--highlight_words",
        "True",
        "--device",
        device,
        "--compute_type",
        compute_type,
    ]
    if language:
        command_preview += ["--language", language]

    model = None
    model_a = None
    metadata = None
    audio = None
    try:
        model = whisperx.load_model(LYRICS_WHISPERX_MODEL, device, compute_type=compute_type)
        audio = whisperx.load_audio(str(audio_path))
        kwargs = {"batch_size": LYRICS_BATCH_SIZE}
        if language:
            kwargs["language"] = language
        raw_result = model.transcribe(audio, **kwargs)
        result_segments = raw_result.get("segments", [])
        detected_language = raw_result.get("language") or language
        extra: Dict[str, Any] = {"device": device, "compute_type": compute_type}

        try:
            model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
            aligned_result = whisperx.align(
                result_segments,
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            result_segments = aligned_result.get("segments", result_segments)
            extra["alignment"] = "whisperx"
        except Exception as exc:  # noqa: BLE001
            extra["alignment_warning"] = str(exc)

        payload = payload_from_segments(
            source="whisperx",
            language=detected_language,
            raw_segments=result_segments,
            raw_text=" ".join(_coerce_text(seg.get("text")) for seg in result_segments),
            extra=extra,
        )
        return payload, {"device": device, "compute_type": compute_type, "language": detected_language}, command_preview
    finally:
        model_a = None
        metadata = None
        model = None
        audio = None
        release_inference_memory()


def align_reference_with_stable(audio_path: Path, reference_path: Path, language: Optional[str]) -> tuple[dict, dict, list[str]]:
    import stable_whisper

    device = choose_device()
    command_preview = [
        "stable-ts",
        str(audio_path),
        "--align",
        str(reference_path),
        "--language",
        language or "<required>",
        "-o",
        "lyrics_aligned.json",
    ]
    if not language:
        raise ValueError("Reference alignment needs lyrics_language or a previous transcription with detected language.")

    model = None
    result = None
    raw_json_path = audio_path.parent / f".{audio_path.stem}.stable_align_temp.json"
    try:
        model = stable_whisper.load_model(LYRICS_ALIGN_MODEL, device=device)
        text = reference_path.read_text(encoding="utf-8")
        result = model.align(str(audio_path), text, language=language)

        result.save_as_json(str(raw_json_path), ensure_ascii=False)
        raw = json.loads(raw_json_path.read_text(encoding="utf-8"))

        payload = payload_from_segments(
            source="stable-ts-align",
            language=language,
            raw_segments=raw.get("segments", []),
            raw_text=text,
            extra={"device": device, "model": LYRICS_ALIGN_MODEL, "reference_path": str(reference_path)},
        )
        metadata = {
            "device": device,
            "language": language,
            "reference_path": str(reference_path),
        }
        return payload, metadata, command_preview
    finally:
        raw_json_path.unlink(missing_ok=True)
        result = None
        model = None
        release_inference_memory()


def planned_outputs(prefix: str, output_root: Path) -> dict[str, str]:
    base = output_root / prefix
    return {
        f"{prefix}_json": str(base.with_suffix(".json")),
        f"{prefix}_txt": str(base.with_suffix(".txt")),
        f"{prefix}_lrc": str(base.with_suffix(".lrc")),
        f"{prefix}_srt": str(base.with_suffix(".srt")),
        "lyrics_json": str(base.with_suffix(".json")),
        "lyrics_txt": str(base.with_suffix(".txt")),
        "lyrics_lrc": str(base.with_suffix(".lrc")),
        "lyrics_srt": str(base.with_suffix(".srt")),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "lyrics-worker", "dry_run": DRY_RUN}


@app.get("/capabilities")
def capabilities() -> dict:
    return {
        "service": "lyrics-worker",
        "tasks": ["transcribe_wordlevel", "align_reference_lyrics"],
        "primary_models": {
            "transcription": LYRICS_WHISPERX_MODEL,
            "alignment": LYRICS_ALIGN_MODEL,
        },
        "outputs": ["json", "txt", "lrc", "srt"],
    }


@app.post("/run")
def run(request: RunRequest) -> dict:
    output_root = output_root_for(request)

    if request.task == "transcribe_wordlevel":
        command_preview = [
            "whisperx",
            request.input_path,
            "--model",
            LYRICS_WHISPERX_MODEL,
            "--output_format",
            "json",
        ]
        named_outputs = planned_outputs("lyrics_wordlevel", output_root)
        if request.dry_run:
            result = RunResult(
                service="lyrics-worker",
                status="dry_run_completed",
                task=request.task,
                target_stem=request.target_stem,
                command=command_preview,
                output_files=list(named_outputs.values()),
                named_outputs=named_outputs,
                metadata={"language": request.lyrics_language},
                notes="Lyrics worker is wired for real local transcription. This dry-run only previews the WhisperX path and output files.",
            )
            return result.model_dump()

        payload, metadata, command_preview = run_lyrics_subprocess(
            "transcribe",
            {
                "audio_path": request.input_path,
                "language": request.lyrics_language,
            },
        )
        named_outputs = write_payload_files("lyrics_wordlevel", payload, output_root)
        result = RunResult(
            service="lyrics-worker",
            status="completed",
            task=request.task,
            target_stem=request.target_stem,
            command=command_preview,
            output_files=list(named_outputs.values()),
            named_outputs=named_outputs,
            metadata=metadata,
            notes="WhisperX transcription completed. This is the best automatic pass, but exact 1:1 lyrics still depend on the song and language.",
        )
        return result.model_dump()

    if request.task == "align_reference_lyrics":
        reference_path = request.lyrics_reference_path
        named_outputs = planned_outputs("lyrics_aligned", output_root)
        language = infer_language(request)
        command_preview = [
            "stable-ts",
            request.input_path,
            "--align",
            reference_path or "<missing_reference>",
            "--language",
            language or "<required>",
            "-o",
            str(output_root / "lyrics_aligned.json"),
        ]

        if not reference_path:
            result = RunResult(
                service="lyrics-worker",
                status="skipped_missing_reference",
                task=request.task,
                target_stem=request.target_stem,
                command=command_preview,
                output_files=[],
                named_outputs={},
                metadata={},
                notes="No lyrics_reference_path was provided, so exact-text alignment was skipped.",
            )
            return result.model_dump()

        if request.dry_run:
            result = RunResult(
                service="lyrics-worker",
                status="dry_run_completed",
                task=request.task,
                target_stem=request.target_stem,
                command=command_preview,
                output_files=list(named_outputs.values()),
                named_outputs=named_outputs,
                metadata={"language": language, "reference_path": reference_path},
                notes="Stable-ts alignment preview. Real run will align the provided reference text to the vocals audio at word level.",
            )
            return result.model_dump()

        payload, metadata, command_preview = run_lyrics_subprocess(
            "align",
            {
                "audio_path": request.input_path,
                "reference_path": reference_path,
                "language": language,
            },
        )
        named_outputs = write_payload_files("lyrics_aligned", payload, output_root)
        result = RunResult(
            service="lyrics-worker",
            status="completed",
            task=request.task,
            target_stem=request.target_stem,
            command=command_preview,
            output_files=list(named_outputs.values()),
            named_outputs=named_outputs,
            metadata=metadata,
            notes="Reference lyrics aligned with stable-ts. This is the closest path to exact text + precise timings inside a local stack.",
        )
        return result.model_dump()

    result = RunResult(
        service="lyrics-worker",
        status="unsupported_task",
        task=request.task,
        target_stem=request.target_stem,
        command=[],
        output_files=[],
        named_outputs={},
        metadata={},
        notes=f"Unsupported lyrics task: {request.task}",
    )
    return result.model_dump()
