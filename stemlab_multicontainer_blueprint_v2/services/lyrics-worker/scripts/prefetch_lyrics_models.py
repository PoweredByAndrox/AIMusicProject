from __future__ import annotations

import json
import os
import wave
from pathlib import Path

import whisper
import whisperx


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def write_silence_wav(path: Path, duration_sec: float = 0.6, sample_rate: int = 16000) -> None:
    frame_count = max(1, int(duration_sec * sample_rate))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)


def cache_root() -> Path:
    explicit = os.getenv("LYRICS_CACHE_ROOT")
    if explicit:
        return Path(explicit)
    return Path(os.getenv("MODELS_ROOT", "/app/models")) / "lyrics-worker-cache"


def prefetch_signature() -> dict:
    return {
        "whisperx_model": os.getenv("LYRICS_WHISPERX_MODEL", "large-v3"),
        "align_model": os.getenv("LYRICS_ALIGN_MODEL", "large-v3"),
        "language": os.getenv("LYRICS_PREFETCH_LANGUAGE", "en"),
    }


def main() -> int:
    if not env_bool("LYRICS_PREFETCH_ON_START", True):
        print("lyrics prefetch disabled")
        return 0

    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    marker_path = root / "prefetch_state.json"
    signature = prefetch_signature()

    if marker_path.exists():
        try:
            current = json.loads(marker_path.read_text(encoding="utf-8"))
            if current == signature:
                print("lyrics prefetch cache already warm")
                return 0
        except Exception:
            pass

    language = signature["language"]
    whisperx_model_name = signature["whisperx_model"]
    align_model_name = signature["align_model"]
    silence_path = root / "warmup" / "silence.wav"
    write_silence_wav(silence_path)

    print(f"prefetch whisperx model: {whisperx_model_name}")
    model = whisperx.load_model(whisperx_model_name, "cpu", compute_type="int8")
    audio = whisperx.load_audio(str(silence_path))
    result = model.transcribe(audio, batch_size=1, language=language)
    del model

    detected_language = result.get("language") or language
    print(f"prefetch whisperx align model: {detected_language}")
    align_model, metadata = whisperx.load_align_model(language_code=detected_language, device="cpu")
    try:
        whisperx.align(
            result.get("segments", []),
            align_model,
            metadata,
            audio,
            "cpu",
            return_char_alignments=False,
        )
    except Exception as exc:
        print(f"lyrics prefetch alignment warmup skipped: {exc}")
    del align_model

    print(f"prefetch whisper/stable-ts model: {align_model_name}")
    whisper.load_model(align_model_name, device="cpu")

    marker_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
    print("lyrics prefetch completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
