from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import librosa
import numpy as np
import soundfile as sf

from worker.instrument_melody_profile import FEATURE_SAMPLE_RATE, NOTE_NAMES, separate_track_stems


DEFAULT_SEPARATOR_MODEL = "htdemucs_6s"
DEFAULT_CACHE_ROOT = Path(os.getenv("MELODY_ALIGNMENT_CACHE_DIR", "/app/storage/tmp/melody_alignment"))
DEFAULT_COMPARE_STEMS = ("other", "bass")
DEFAULT_HOP_SEC = 0.25
DEFAULT_MIN_CONFIDENCE = 0.4
DEFAULT_MIN_DURATION_SEC = 0.12
DEFAULT_BEAT_HOP_LENGTH = 512
DEFAULT_BEAT_START_BPM = 140.0
DEFAULT_ALIGNMENT_PIPELINE_VERSION = "v2_orchestral_proxy"


def _audio_signature(audio_path: Path, separator_model: str) -> str:
    stat = audio_path.stat()
    fingerprint = "|".join(
        [
            str(audio_path.resolve()),
            str(int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))),
            str(int(stat.st_size)),
            separator_model,
            DEFAULT_ALIGNMENT_PIPELINE_VERSION,
        ]
    )
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()


def _track_cache_dir(audio_path: Path, separator_model: str, cache_root: Path) -> Path:
    return cache_root / f"{audio_path.stem}_{_audio_signature(audio_path, separator_model)}"


def _ensure_stem_wavs(
    audio_path: str | Path,
    *,
    stems: Iterable[str],
    separator_model: str,
    device: str | None,
    cache_root: Path,
) -> dict[str, Path]:
    resolved_audio_path = Path(audio_path)
    track_cache_dir = _track_cache_dir(resolved_audio_path, separator_model, cache_root)
    stem_dir = track_cache_dir / "stems"
    stem_dir.mkdir(parents=True, exist_ok=True)

    stem_paths = {stem_name: stem_dir / f"{stem_name}.wav" for stem_name in stems}
    if all(path.exists() for path in stem_paths.values()):
        return stem_paths

    separated = separate_track_stems(resolved_audio_path, model_name=separator_model, device=device)
    for stem_name, target_path in stem_paths.items():
        if stem_name not in separated:
            raise KeyError(f"Stem {stem_name!r} missing from separator output: {sorted(separated.keys())}")
        if not target_path.exists():
            sf.write(target_path, np.asarray(separated[stem_name], dtype=np.float32), FEATURE_SAMPLE_RATE)
    return stem_paths


def _extract_note_events_subprocess(stem_paths: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    script = """
import contextlib
import io
import json
import sys
from pathlib import Path
import basic_pitch
import basic_pitch.inference as basic_pitch_inference
from basic_pitch.inference import predict

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
DEFAULT_MODEL_ROOT = Path(basic_pitch.__file__).resolve().parent / 'saved_models' / 'icassp_2022'
DEFAULT_TFLITE_MODEL_PATH = DEFAULT_MODEL_ROOT / 'nmp.tflite'
DEFAULT_ONNX_MODEL_PATH = DEFAULT_MODEL_ROOT / 'nmp.onnx'
MODEL_PATH = DEFAULT_TFLITE_MODEL_PATH

try:
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        _original_model_init = basic_pitch_inference.Model.__init__

        def _patched_model_init(self, model_path):
            model_path = Path(model_path)
            if str(model_path).endswith('.onnx'):
                self.model_type = basic_pitch_inference.Model.MODEL_TYPES.ONNX
                self.model = ort.InferenceSession(
                    str(model_path),
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                )
                return
            return _original_model_init(self, model_path)

        basic_pitch_inference.Model.__init__ = _patched_model_init
        MODEL_PATH = DEFAULT_ONNX_MODEL_PATH
except Exception:
    MODEL_PATH = DEFAULT_TFLITE_MODEL_PATH

def transcribe(path):
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, note_events = predict(path, model_or_model_path=str(MODEL_PATH))
    payload = []
    for start, end, pitch_midi, confidence, pitch_bend in note_events:
        midi = int(round(float(pitch_midi)))
        octave = (midi // 12) - 1
        payload.append({
            'start': round(float(start), 4),
            'end': round(float(end), 4),
            'midi': midi,
            'note_name': f"{NOTE_NAMES[midi % 12]}{octave}",
            'note_class': NOTE_NAMES[midi % 12],
            'confidence': round(float(confidence), 4),
        })
    payload.sort(key=lambda item: (item['start'], item['midi']))
    return payload

result = {}
args = sys.argv[1:]
for index in range(0, len(args), 2):
    stem_name = args[index]
    path = args[index + 1]
    result[stem_name] = transcribe(path)
print(json.dumps(result))
"""
    command = [sys.executable, "-c", script]
    for stem_name, stem_path in stem_paths.items():
        command.extend([stem_name, str(stem_path)])
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or f"exit_code={exc.returncode}"
        raise RuntimeError(f"basic_pitch note-event subprocess failed: {detail}") from exc
    stdout = completed.stdout.strip()
    if not stdout:
        stderr = completed.stderr.strip()
        detail = stderr or "empty stdout"
        raise RuntimeError(f"basic_pitch note-event subprocess returned no JSON output: {detail}")
    return json.loads(stdout)


def _ensure_other_orchestral_proxy_wav(other_stem_path: Path, *, cache_root: Path) -> Path:
    proxy_path = cache_root / "stems" / "other_orchestral_proxy.wav"
    if proxy_path.exists():
        return proxy_path

    audio_signal, sample_rate = librosa.load(str(other_stem_path), sr=FEATURE_SAMPLE_RATE, mono=True)
    harmonic_signal = librosa.effects.harmonic(audio_signal, margin=6.0)

    stft_matrix = librosa.stft(harmonic_signal, n_fft=2048, hop_length=512)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    frequency_weights = np.ones_like(frequencies, dtype=np.float32)
    frequency_weights[frequencies < 170.0] = 0.18
    frequency_weights[(frequencies >= 170.0) & (frequencies < 260.0)] = 0.45
    frequency_weights[(frequencies >= 260.0) & (frequencies <= 4200.0)] = 1.0
    frequency_weights[(frequencies > 4200.0) & (frequencies <= 6500.0)] = 0.55
    frequency_weights[frequencies > 6500.0] = 0.22
    filtered_stft = stft_matrix * frequency_weights[:, None]
    filtered_signal = librosa.istft(filtered_stft, hop_length=512, length=len(harmonic_signal))

    peak = float(np.max(np.abs(filtered_signal))) if filtered_signal.size else 0.0
    if peak > 0.0:
        filtered_signal = 0.95 * (filtered_signal / peak)

    proxy_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(proxy_path, np.asarray(filtered_signal, dtype=np.float32), sample_rate)
    return proxy_path


def load_or_create_note_events(
    audio_path: str | Path,
    *,
    stems: Iterable[str] = DEFAULT_COMPARE_STEMS,
    separator_model: str = DEFAULT_SEPARATOR_MODEL,
    device: str | None = None,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    resolved_audio_path = Path(audio_path)
    if not resolved_audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved_audio_path}")

    resolved_cache_root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    resolved_cache_root.mkdir(parents=True, exist_ok=True)
    track_cache_dir = _track_cache_dir(resolved_audio_path, separator_model, resolved_cache_root)
    events_path = track_cache_dir / "note_events.json"
    if events_path.exists():
        return json.loads(events_path.read_text(encoding="utf-8"))

    requested_stems = tuple(stems)
    stem_paths = _ensure_stem_wavs(
        resolved_audio_path,
        stems=requested_stems,
        separator_model=separator_model,
        device=device,
        cache_root=resolved_cache_root,
    )
    transcription_paths = dict(stem_paths)
    other_proxy_path: Path | None = None
    if "other" in stem_paths:
        other_proxy_path = _ensure_other_orchestral_proxy_wav(
            stem_paths["other"],
            cache_root=track_cache_dir,
        )
        transcription_paths["other_orchestral_proxy"] = other_proxy_path

    note_events = _extract_note_events_subprocess(transcription_paths)

    payload = {
        "audio_path": str(resolved_audio_path),
        "separator_model": separator_model,
        "sample_rate": FEATURE_SAMPLE_RATE,
        "stems": {},
    }
    for stem_name, stem_path in stem_paths.items():
        info = sf.info(str(stem_path))
        payload["stems"][stem_name] = {
            "audio_path": str(stem_path),
            "duration_sec": round(float(info.frames / info.samplerate), 4),
            "note_events": note_events.get(stem_name, []),
        }
        if stem_name == "other" and other_proxy_path is not None:
            payload["stems"][stem_name]["orchestral_proxy_audio_path"] = str(other_proxy_path)
            payload["stems"][stem_name]["orchestral_note_events"] = note_events.get("other_orchestral_proxy", [])

    track_cache_dir.mkdir(parents=True, exist_ok=True)
    events_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_or_create_beat_grid(
    audio_path: str | Path,
    *,
    separator_model: str = DEFAULT_SEPARATOR_MODEL,
    cache_root: str | Path | None = None,
    hop_length: int = DEFAULT_BEAT_HOP_LENGTH,
    start_bpm: float = DEFAULT_BEAT_START_BPM,
) -> dict[str, Any]:
    resolved_audio_path = Path(audio_path)
    if not resolved_audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved_audio_path}")

    resolved_cache_root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    resolved_cache_root.mkdir(parents=True, exist_ok=True)
    track_cache_dir = _track_cache_dir(resolved_audio_path, separator_model, resolved_cache_root)
    beat_grid_path = track_cache_dir / "beat_grid.json"
    if beat_grid_path.exists():
        return json.loads(beat_grid_path.read_text(encoding="utf-8"))

    audio_signal, sample_rate = librosa.load(str(resolved_audio_path), sr=FEATURE_SAMPLE_RATE, mono=True)
    duration_sec = round(float(len(audio_signal) / sample_rate), 4)
    onset_env = librosa.onset.onset_strength(
        y=audio_signal,
        sr=sample_rate,
        hop_length=hop_length,
    )
    estimated_tempo, beat_times = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop_length,
        start_bpm=float(start_bpm),
        units="time",
    )

    beat_grid = sorted(
        {
            round(float(beat), 4)
            for beat in np.atleast_1d(beat_times)
            if 0.0 <= float(beat) <= duration_sec
        }
    )
    payload = {
        "audio_path": str(resolved_audio_path),
        "duration_sec": duration_sec,
        "tempo_bpm": round(float(np.atleast_1d(estimated_tempo).reshape(-1)[0]), 3) if np.atleast_1d(estimated_tempo).size else None,
        "beat_times": beat_grid,
    }
    track_cache_dir.mkdir(parents=True, exist_ok=True)
    beat_grid_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _normalize_window(window: tuple[float, float] | None, duration_sec: float) -> tuple[float, float]:
    if window is None:
        return 0.0, float(duration_sec)

    start = max(0.0, min(float(duration_sec), float(window[0])))
    end = max(start, min(float(duration_sec), float(window[1])))
    return round(start, 4), round(end, 4)


def _filter_events_to_window(
    note_events: list[dict[str, Any]],
    window_start: float,
    window_end: float,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for event in note_events:
        event_start = float(event["start"])
        event_end = float(event["end"])
        if event_end <= window_start or event_start >= window_end:
            continue

        clipped_start = max(window_start, event_start) - window_start
        clipped_end = min(window_end, event_end) - window_start
        if clipped_end <= clipped_start:
            continue

        item = dict(event)
        item["start"] = round(float(clipped_start), 4)
        item["end"] = round(float(clipped_end), 4)
        filtered.append(item)
    return filtered


def _window_beat_boundaries(
    beat_times: Sequence[float],
    window_start: float,
    window_end: float,
    *,
    fallback_hop: float,
) -> list[float]:
    duration_sec = max(0.0, float(window_end) - float(window_start))
    if duration_sec <= 0.0:
        return [0.0, 0.0]

    internal = [
        round(float(beat) - float(window_start), 4)
        for beat in beat_times
        if float(window_start) < float(beat) < float(window_end)
    ]
    boundaries = sorted({0.0, *internal, round(duration_sec, 4)})
    if len(boundaries) >= 3:
        return boundaries

    step = max(0.1, float(fallback_hop))
    generated = list(np.arange(0.0, duration_sec, step, dtype=np.float32))
    boundaries = sorted({round(float(value), 4) for value in generated} | {0.0, round(duration_sec, 4)})
    if len(boundaries) == 1:
        boundaries.append(round(duration_sec, 4))
    return boundaries


def _build_frame_sequence(
    note_events: list[dict[str, Any]],
    duration_sec: float,
    *,
    hop_sec: float,
    stem_name: str,
    min_confidence: float,
    min_duration_sec: float,
) -> list[int | None]:
    filtered_events = [
        event for event in note_events
        if float(event["confidence"]) >= min_confidence
        and (float(event["end"]) - float(event["start"])) >= min_duration_sec
    ]
    if not filtered_events:
        frame_count = max(1, int(math.ceil(duration_sec / hop_sec)))
        return [None] * frame_count

    frame_count = max(1, int(math.ceil(duration_sec / hop_sec)))
    sequence: list[int | None] = []
    for frame_index in range(frame_count):
        frame_time = frame_index * hop_sec
        active = [
            event for event in filtered_events
            if float(event["start"]) <= frame_time < float(event["end"])
        ]
        if not active:
            sequence.append(None)
            continue

        if stem_name == "bass":
            chosen = min(active, key=lambda event: (int(event["midi"]), -float(event["confidence"])))
        else:
            chosen = max(active, key=lambda event: (int(event["midi"]), float(event["confidence"])))
        sequence.append(int(chosen["midi"]))
    return sequence


def _build_beat_vector_sequence(
    note_events: list[dict[str, Any]],
    beat_boundaries: Sequence[float],
    *,
    vector_mode: str,
    min_confidence: float,
    min_duration_sec: float,
) -> tuple[list[np.ndarray], list[float]]:
    filtered_events = [
        event for event in note_events
        if float(event["confidence"]) >= min_confidence
        and (float(event["end"]) - float(event["start"])) >= min_duration_sec
    ]

    vector_size = 12 if vector_mode == "pitch_class" else 128
    vectors: list[np.ndarray] = []
    densities: list[float] = []
    if len(beat_boundaries) < 2:
        return [np.zeros(vector_size, dtype=np.float32)], [0.0]

    for start, end in zip(beat_boundaries[:-1], beat_boundaries[1:]):
        vector = np.zeros(vector_size, dtype=np.float32)
        if end <= start:
            vectors.append(vector)
            densities.append(0.0)
            continue

        for event in filtered_events:
            overlap = min(float(event["end"]), float(end)) - max(float(event["start"]), float(start))
            if overlap <= 0.0:
                continue

            midi = int(event["midi"])
            index = midi % 12 if vector_mode == "pitch_class" else max(0, min(127, midi))
            vector[index] += float(overlap) * float(event["confidence"])

        density = float(vector.sum())
        if density > 0.0:
            vector /= density
        vectors.append(vector)
        densities.append(density)
    return vectors, densities


def _dominant_pitch_classes(vectors: Sequence[np.ndarray], densities: Sequence[float]) -> list[int | None]:
    sequence: list[int | None] = []
    for vector, density in zip(vectors, densities):
        if density <= 0.0 or float(vector.sum()) <= 0.0:
            sequence.append(None)
            continue
        sequence.append(int(np.argmax(vector)))
    return sequence


def _compress_pitch_classes(sequence: Sequence[int | None]) -> list[int]:
    compressed: list[int] = []
    for item in sequence:
        if item is None:
            continue
        if not compressed or compressed[-1] != item:
            compressed.append(int(item))
    return compressed


def _signed_shortest_interval(previous_pitch_class: int, current_pitch_class: int) -> int:
    raw = (int(current_pitch_class) - int(previous_pitch_class)) % 12
    if raw == 6:
        return 6
    return raw if raw < 6 else raw - 12


def _interval_sequence(pitch_classes: Sequence[int]) -> list[int]:
    if len(pitch_classes) < 2:
        return []
    return [
        _signed_shortest_interval(previous, current)
        for previous, current in zip(pitch_classes[:-1], pitch_classes[1:])
    ]


def _dtw_similarity(
    seq_a: Sequence[Any],
    seq_b: Sequence[Any],
    cost_fn: Callable[[Any, Any], float],
) -> float:
    n = len(seq_a)
    m = len(seq_b)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    steps = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0
    steps[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            local_cost = float(cost_fn(seq_a[i - 1], seq_b[j - 1]))
            candidates = (
                (cost[i - 1, j], steps[i - 1, j]),
                (cost[i, j - 1], steps[i, j - 1]),
                (cost[i - 1, j - 1], steps[i - 1, j - 1]),
            )
            best_cost, best_steps = min(candidates, key=lambda item: (item[0], item[1]))
            cost[i, j] = local_cost + best_cost
            steps[i, j] = best_steps + 1.0

    normalized_cost = float(cost[n, m] / max(steps[n, m], 1.0))
    return round(max(0.0, 1.0 - normalized_cost), 4)


def _sequence_similarity(
    seq_a: Sequence[Any],
    seq_b: Sequence[Any],
    cost_fn: Callable[[Any, Any], float],
) -> float:
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0
    return _dtw_similarity(seq_a, seq_b, cost_fn)


def _pitch_class_cost(a: int | None, b: int | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1.0
    distance = abs((a % 12) - (b % 12))
    circular = min(distance, 12 - distance)
    return float(circular / 6.0)


def _note_name_cost(a: int | None, b: int | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1.0
    return float(min(abs(a - b), 24) / 24.0)


def _interval_cost(a: int, b: int) -> float:
    return float(min(abs(int(a) - int(b)), 12) / 12.0)


def _vector_cosine_cost(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 and norm_b == 0.0:
        return 0.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - max(0.0, similarity)


def _shift_pitch_classes(sequence: Sequence[int | None], semitone_shift: int) -> list[int | None]:
    shifted: list[int | None] = []
    for item in sequence:
        if item is None:
            shifted.append(None)
        else:
            shifted.append(int(item + semitone_shift))
    return shifted


def _shift_pitch_class_vectors(sequence: Sequence[np.ndarray], semitone_shift: int) -> list[np.ndarray]:
    shift = int(semitone_shift) % 12
    if shift == 0:
        return [np.asarray(item, dtype=np.float32) for item in sequence]
    return [np.roll(np.asarray(item, dtype=np.float32), shift).astype(np.float32, copy=False) for item in sequence]


def _top_notes(events: list[dict[str, Any]], *, limit: int) -> dict[str, list[str]]:
    note_names: dict[str, int] = {}
    note_classes: dict[str, int] = {}
    for event in events:
        note_names[str(event["note_name"])] = note_names.get(str(event["note_name"]), 0) + 1
        note_classes[str(event["note_class"])] = note_classes.get(str(event["note_class"]), 0) + 1
    return {
        "top_note_names": [name for name, _ in sorted(note_names.items(), key=lambda item: (-item[1], item[0]))[:limit]],
        "top_note_classes": [name for name, _ in sorted(note_classes.items(), key=lambda item: (-item[1], item[0]))[:limit]],
    }


def _note_classes_from_pitch_classes(pitch_classes: Sequence[int]) -> list[str]:
    return [NOTE_NAMES[int(pitch_class) % 12] for pitch_class in pitch_classes]


def _pitch_classes_from_note_classes(note_classes: Sequence[str]) -> list[int]:
    return [NOTE_NAMES.index(str(note_class)) for note_class in note_classes if str(note_class) in NOTE_NAMES]


def _build_beat_candidate_sequence(
    note_events: list[dict[str, Any]],
    beat_boundaries: Sequence[float],
    *,
    stem_name: str,
    min_confidence: float,
    min_duration_sec: float,
    max_candidates: int = 4,
) -> list[list[tuple[int, float]]]:
    filtered_events = [
        event for event in note_events
        if float(event["confidence"]) >= min_confidence
        and (float(event["end"]) - float(event["start"])) >= min_duration_sec
    ]
    candidates_per_beat: list[list[tuple[int, float]]] = []
    if len(beat_boundaries) < 2:
        return [[]]

    for start, end in zip(beat_boundaries[:-1], beat_boundaries[1:]):
        weights: dict[int, float] = {}
        for event in filtered_events:
            overlap = min(float(event["end"]), float(end)) - max(float(event["start"]), float(start))
            if overlap <= 0.0:
                continue

            midi = int(event["midi"])
            weight = float(overlap) * float(event["confidence"])
            if stem_name == "other":
                if midi < 48:
                    weight *= 0.55
                elif midi >= 72:
                    weight *= 1.1
            elif stem_name == "bass":
                if midi > 60:
                    weight *= 0.45
                elif midi <= 48:
                    weight *= 1.1
            weights[midi] = weights.get(midi, 0.0) + weight

        if stem_name == "bass":
            ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
        else:
            ranked = sorted(weights.items(), key=lambda item: (-item[1], -item[0]))
        candidates_per_beat.append([(int(midi), round(float(weight), 6)) for midi, weight in ranked[:max_candidates]])
    return candidates_per_beat


def _melodic_event_salience(event: dict[str, Any], *, stem_name: str) -> float:
    midi = int(event["midi"])
    duration = max(0.0, float(event["end"]) - float(event["start"]))
    confidence = max(0.0, min(1.0, float(event["confidence"])))

    if stem_name == "other":
        register_bonus = max(0.0, min(midi - 55, 24) / 24.0)
        sustain_bonus = max(0.0, min(duration, 0.9) / 0.9)
        return round(confidence * 2.2 + sustain_bonus * 0.9 + register_bonus * 0.85, 6)

    register_bonus = max(0.0, min(60 - min(midi, 60), 24) / 24.0)
    sustain_bonus = max(0.0, min(duration, 0.75) / 0.75)
    return round(confidence * 2.0 + sustain_bonus * 0.8 + register_bonus * 0.75, 6)


def _filter_melodic_source_events(
    note_events: list[dict[str, Any]],
    *,
    stem_name: str,
    min_confidence: float,
    min_duration_sec: float,
) -> list[dict[str, Any]]:
    confidence_floor = max(min_confidence, 0.32 if stem_name == "other" else 0.28)
    duration_floor = max(min_duration_sec, 0.16 if stem_name == "other" else 0.12)
    deduped: dict[tuple[int, int, int], dict[str, Any]] = {}

    for event in note_events:
        start = round(float(event["start"]), 4)
        end = round(float(event["end"]), 4)
        duration = round(max(0.0, end - start), 4)
        confidence = round(float(event["confidence"]), 4)
        midi = int(event["midi"])
        if confidence < confidence_floor or duration < duration_floor:
            continue
        if stem_name == "other" and midi < 52:
            continue
        if stem_name == "bass" and midi > 67:
            continue

        candidate = {
            "start": start,
            "end": end,
            "duration": duration,
            "midi": midi,
            "note_name": str(event["note_name"]),
            "note_class": str(event["note_class"]),
            "confidence": confidence,
        }
        candidate["salience"] = _melodic_event_salience(candidate, stem_name=stem_name)
        dedupe_key = (int(round(start * 100)), int(round(end * 100)), midi)
        previous = deduped.get(dedupe_key)
        if previous is None or float(candidate["salience"]) > float(previous["salience"]):
            deduped[dedupe_key] = candidate

    return sorted(deduped.values(), key=lambda item: (float(item["start"]), float(item["end"]), int(item["midi"])))


def _melodic_transition_score(previous_midi: int | None, current_midi: int | None, *, stem_name: str) -> float:
    if current_midi is None:
        return -0.08

    if stem_name == "other":
        register_bonus = max(0.0, min(current_midi - 50, 26) / 52.0)
    else:
        register_bonus = max(0.0, min(60 - current_midi, 24) / 48.0)

    score = register_bonus
    if previous_midi is None:
        return score

    interval = abs(int(current_midi) - int(previous_midi))
    score -= min(interval, 24) / 20.0
    if interval <= 2:
        score += 0.1
    elif interval <= 5:
        score += 0.05
    return score


def _melodic_event_transition_score(
    previous_event: dict[str, Any],
    current_event: dict[str, Any],
    *,
    stem_name: str,
) -> float:
    gap_sec = float(current_event["start"]) - float(previous_event["end"])
    if gap_sec < -0.32:
        return -float("inf")

    score = _melodic_transition_score(
        int(previous_event["midi"]),
        int(current_event["midi"]),
        stem_name=stem_name,
    )
    if gap_sec < 0.0:
        score -= min(abs(gap_sec), 0.24) * 2.1
    elif gap_sec <= 0.45:
        score += 0.14 - gap_sec * 0.18
    else:
        score -= min(gap_sec, 1.5) * 0.38

    if current_event["note_class"] == previous_event["note_class"]:
        score += 0.05
    return score


def _extract_melodic_event_chain(
    note_events: list[dict[str, Any]],
    *,
    stem_name: str,
    min_confidence: float,
    min_duration_sec: float,
) -> list[dict[str, Any]]:
    filtered_events = _filter_melodic_source_events(
        note_events,
        stem_name=stem_name,
        min_confidence=min_confidence,
        min_duration_sec=min_duration_sec,
    )
    if not filtered_events:
        return []

    max_gap_sec = 1.6 if stem_name == "other" else 1.25
    max_predecessors = 48
    scores: list[float] = []
    backpointers: list[int] = []

    for current_index, current_event in enumerate(filtered_events):
        base_score = float(current_event["salience"]) + _melodic_transition_score(
            None,
            int(current_event["midi"]),
            stem_name=stem_name,
        )
        best_score = base_score
        best_previous_index = -1

        predecessor_budget = 0
        for previous_index in range(current_index - 1, -1, -1):
            previous_event = filtered_events[previous_index]
            if predecessor_budget >= max_predecessors:
                break
            predecessor_budget += 1

            if (
                float(current_event["start"]) - float(previous_event["start"]) > max_gap_sec + 0.85
                and float(current_event["start"]) - float(previous_event["end"]) > max_gap_sec
            ):
                break

            transition_score = _melodic_event_transition_score(previous_event, current_event, stem_name=stem_name)
            if not math.isfinite(transition_score):
                continue

            candidate_score = scores[previous_index] + float(current_event["salience"]) + transition_score
            if candidate_score > best_score:
                best_score = candidate_score
                best_previous_index = previous_index

        scores.append(best_score)
        backpointers.append(best_previous_index)

    last_index = max(range(len(scores)), key=lambda index: scores[index])
    melodic_chain: list[dict[str, Any]] = []
    while last_index >= 0:
        melodic_chain.append(filtered_events[last_index])
        last_index = backpointers[last_index]
    melodic_chain.reverse()
    return melodic_chain


def _extract_melodic_layer_from_candidates(
    candidates_per_beat: Sequence[Sequence[tuple[int, float]]],
    *,
    stem_name: str,
) -> list[int | None]:
    if not candidates_per_beat:
        return []

    normalized_candidates: list[list[tuple[int | None, float]]] = []
    for candidates in candidates_per_beat:
        if candidates:
            normalized_candidates.append([(int(midi), float(weight)) for midi, weight in candidates])
        else:
            normalized_candidates.append([(None, 0.0)])

    scores: list[list[float]] = []
    backpointers: list[list[int]] = []
    first_row_scores = []
    for midi, weight in normalized_candidates[0]:
        first_row_scores.append(float(weight) * 1.5 + _melodic_transition_score(None, midi, stem_name=stem_name))
    scores.append(first_row_scores)
    backpointers.append([-1] * len(normalized_candidates[0]))

    for beat_index in range(1, len(normalized_candidates)):
        current_candidates = normalized_candidates[beat_index]
        previous_candidates = normalized_candidates[beat_index - 1]
        current_scores: list[float] = []
        current_backpointers: list[int] = []

        for current_index, (current_midi, current_weight) in enumerate(current_candidates):
            best_score = -float("inf")
            best_previous_index = 0
            for previous_index, (previous_midi, _) in enumerate(previous_candidates):
                transition_score = _melodic_transition_score(previous_midi, current_midi, stem_name=stem_name)
                candidate_score = scores[beat_index - 1][previous_index] + float(current_weight) * 1.5 + transition_score
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_previous_index = previous_index

            current_scores.append(best_score)
            current_backpointers.append(best_previous_index)

        scores.append(current_scores)
        backpointers.append(current_backpointers)

    last_index = max(range(len(scores[-1])), key=lambda index: scores[-1][index])
    melodic_line: list[int | None] = []
    for beat_index in range(len(normalized_candidates) - 1, -1, -1):
        midi, _ = normalized_candidates[beat_index][last_index]
        melodic_line.append(midi)
        previous_index = backpointers[beat_index][last_index]
        last_index = previous_index if previous_index >= 0 else 0
    melodic_line.reverse()
    return melodic_line


def _line_to_pitch_classes(sequence: Sequence[int | None]) -> list[int]:
    return [int(midi) % 12 for midi in sequence if midi is not None]


def _line_to_note_names(sequence: Sequence[int | None]) -> list[str]:
    note_names: list[str] = []
    for midi in sequence:
        if midi is None:
            continue
        octave = (int(midi) // 12) - 1
        note_names.append(f"{NOTE_NAMES[int(midi) % 12]}{octave}")
    return note_names


def _event_chain_to_pitch_classes(sequence: Sequence[dict[str, Any]]) -> list[int]:
    return [int(event["midi"]) % 12 for event in sequence]


def _event_chain_to_note_names(sequence: Sequence[dict[str, Any]]) -> list[str]:
    return [str(event["note_name"]) for event in sequence]


def _segment_melodic_phrases(
    melodic_events: Sequence[dict[str, Any]],
    *,
    min_phrase_notes: int = 4,
    max_phrase_notes: int = 12,
    leap_boundary_semitones: int = 7,
    max_gap_sec: float = 0.8,
    max_phrase_duration_sec: float = 8.0,
) -> list[dict[str, Any]]:
    phrases: list[dict[str, Any]] = []
    current_events: list[dict[str, Any]] = []

    def flush_phrase() -> None:
        nonlocal current_events
        if len(current_events) < min_phrase_notes:
            current_events = []
            return

        notes = [int(event["midi"]) for event in current_events]
        if len(notes) < min_phrase_notes:
            current_events = []
            return

        pitch_classes = [int(note) % 12 for note in notes]
        interval_sequence = _interval_sequence(pitch_classes)
        if len(set(pitch_classes)) < 3 or len(interval_sequence) < 2:
            current_events = []
            return

        phrases.append(
            {
                "start_event_index": int(current_events[0]["event_index"]),
                "end_event_index": int(current_events[-1]["event_index"]),
                "start_sec": round(float(current_events[0]["start"]), 4),
                "end_sec": round(float(current_events[-1]["end"]), 4),
                "midi_sequence": notes,
                "note_names": _event_chain_to_note_names(current_events),
                "note_classes": _note_classes_from_pitch_classes(pitch_classes),
                "interval_sequence": interval_sequence,
            }
        )
        current_events = []

    previous_note: int | None = None
    previous_event: dict[str, Any] | None = None
    for event_index, event in enumerate(melodic_events):
        note_value = int(event["midi"])
        if previous_note is not None and note_value == previous_note:
            continue

        should_split = False
        if previous_event is not None:
            gap_sec = float(event["start"]) - float(previous_event["end"])
            interval = abs(note_value - previous_note)
            phrase_duration = float(event["end"]) - float(current_events[0]["start"]) if current_events else 0.0
            if gap_sec > max_gap_sec:
                should_split = True
            if len(current_events) >= min_phrase_notes and interval >= leap_boundary_semitones:
                should_split = True
            if len(current_events) >= max_phrase_notes:
                should_split = True
            if phrase_duration >= max_phrase_duration_sec:
                should_split = True

        if should_split:
            flush_phrase()

        current_events.append(
            {
                "event_index": event_index,
                "start": float(event["start"]),
                "end": float(event["end"]),
                "midi": note_value,
                "note_name": str(event["note_name"]),
            }
        )
        previous_note = note_value
        previous_event = event

    flush_phrase()
    return phrases


def _build_phrase_alignment(
    phrases_a: Sequence[dict[str, Any]],
    phrases_b: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not phrases_a and not phrases_b:
        return {
            "similarity": 1.0,
            "note_class_similarity": 1.0,
            "interval_similarity": 1.0,
            "phrase_index_a": None,
            "phrase_index_b": None,
            "phrase_a": None,
            "phrase_b": None,
        }

    best: dict[str, Any] = {
        "similarity": 0.0,
        "note_class_similarity": 0.0,
        "interval_similarity": 0.0,
        "phrase_index_a": None,
        "phrase_index_b": None,
        "phrase_a": None,
        "phrase_b": None,
    }

    for phrase_index_a, phrase_a in enumerate(phrases_a):
        for phrase_index_b, phrase_b in enumerate(phrases_b):
            note_class_similarity = _sequence_similarity(
                _pitch_classes_from_note_classes(phrase_a["note_classes"]),
                _pitch_classes_from_note_classes(phrase_b["note_classes"]),
                _pitch_class_cost,
            )
            interval_similarity = _sequence_similarity(
                phrase_a["interval_sequence"],
                phrase_b["interval_sequence"],
                _interval_cost,
            )
            similarity = round(0.6 * note_class_similarity + 0.4 * interval_similarity, 4)

            if similarity > best["similarity"]:
                best = {
                    "similarity": similarity,
                    "note_class_similarity": round(float(note_class_similarity), 4),
                    "interval_similarity": round(float(interval_similarity), 4),
                    "phrase_index_a": int(phrase_index_a),
                    "phrase_index_b": int(phrase_index_b),
                    "phrase_a": phrase_a,
                    "phrase_b": phrase_b,
                }

    return best


def _build_motif_alignment(
    interval_sequence_a: Sequence[int],
    interval_sequence_b: Sequence[int],
    note_classes_a: Sequence[str],
    note_classes_b: Sequence[str],
    *,
    min_motif_length: int = 4,
    max_motif_length: int = 8,
) -> dict[str, Any]:
    if len(interval_sequence_a) < min_motif_length or len(interval_sequence_b) < min_motif_length:
        return {
            "similarity": 0.0,
            "motif_length": 0,
            "start_index_a": None,
            "start_index_b": None,
            "interval_sequence_a": [],
            "interval_sequence_b": [],
            "note_classes_a": [],
            "note_classes_b": [],
        }

    best: dict[str, Any] = {
        "similarity": 0.0,
        "interval_similarity": 0.0,
        "note_class_similarity": 0.0,
        "motif_length": 0,
        "start_index_a": None,
        "start_index_b": None,
        "interval_sequence_a": [],
        "interval_sequence_b": [],
        "note_classes_a": [],
        "note_classes_b": [],
    }

    max_length = min(max_motif_length, len(interval_sequence_a), len(interval_sequence_b))
    for motif_length in range(max_length, min_motif_length - 1, -1):
        for start_index_a in range(0, len(interval_sequence_a) - motif_length + 1):
            motif_a = list(interval_sequence_a[start_index_a:start_index_a + motif_length])
            note_slice_a = list(note_classes_a[start_index_a:start_index_a + motif_length + 1])
            if len(set(note_slice_a)) < 3:
                continue

            for start_index_b in range(0, len(interval_sequence_b) - motif_length + 1):
                motif_b = list(interval_sequence_b[start_index_b:start_index_b + motif_length])
                note_slice_b = list(note_classes_b[start_index_b:start_index_b + motif_length + 1])
                if len(set(note_slice_b)) < 3:
                    continue

                interval_similarity = _sequence_similarity(motif_a, motif_b, _interval_cost)
                note_class_similarity = _sequence_similarity(
                    _pitch_classes_from_note_classes(note_slice_a),
                    _pitch_classes_from_note_classes(note_slice_b),
                    _pitch_class_cost,
                )
                similarity = round(0.55 * note_class_similarity + 0.45 * interval_similarity, 4)
                if (
                    similarity > best["similarity"]
                    or (
                        similarity == best["similarity"]
                        and motif_length > best["motif_length"]
                    )
                ):
                    best = {
                        "similarity": round(float(similarity), 4),
                        "interval_similarity": round(float(interval_similarity), 4),
                        "note_class_similarity": round(float(note_class_similarity), 4),
                        "motif_length": int(motif_length),
                        "start_index_a": int(start_index_a),
                        "start_index_b": int(start_index_b),
                        "interval_sequence_a": motif_a,
                        "interval_sequence_b": motif_b,
                        "note_classes_a": note_slice_a,
                        "note_classes_b": note_slice_b,
                    }
    return best


def compare_tracks_by_melody_alignment(
    audio_path_a: str | Path,
    audio_path_b: str | Path,
    *,
    separator_model: str = DEFAULT_SEPARATOR_MODEL,
    device: str | None = None,
    cache_root: str | Path | None = None,
    hop_sec: float = DEFAULT_HOP_SEC,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    window_a: tuple[float, float] | None = None,
    window_b: tuple[float, float] | None = None,
) -> dict[str, Any]:
    cache_root_path = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    track_a = load_or_create_note_events(
        audio_path_a,
        stems=DEFAULT_COMPARE_STEMS,
        separator_model=separator_model,
        device=device,
        cache_root=cache_root_path,
    )
    track_b = load_or_create_note_events(
        audio_path_b,
        stems=DEFAULT_COMPARE_STEMS,
        separator_model=separator_model,
        device=device,
        cache_root=cache_root_path,
    )
    beat_grid_a = load_or_create_beat_grid(
        audio_path_a,
        separator_model=separator_model,
        cache_root=cache_root_path,
    )
    beat_grid_b = load_or_create_beat_grid(
        audio_path_b,
        separator_model=separator_model,
        cache_root=cache_root_path,
    )

    resolved_window_a = _normalize_window(window_a, float(beat_grid_a["duration_sec"]))
    resolved_window_b = _normalize_window(window_b, float(beat_grid_b["duration_sec"]))
    window_duration_a = round(resolved_window_a[1] - resolved_window_a[0], 4)
    window_duration_b = round(resolved_window_b[1] - resolved_window_b[0], 4)

    result: dict[str, Any] = {
        "track_a_path": str(Path(audio_path_a)),
        "track_b_path": str(Path(audio_path_b)),
        "separator_model": separator_model,
        "frame_hop_sec": hop_sec,
        "window_a": {
            "start_sec": resolved_window_a[0],
            "end_sec": resolved_window_a[1],
            "duration_sec": window_duration_a,
            "tempo_bpm": beat_grid_a.get("tempo_bpm"),
        },
        "window_b": {
            "start_sec": resolved_window_b[0],
            "end_sec": resolved_window_b[1],
            "duration_sec": window_duration_b,
            "tempo_bpm": beat_grid_b.get("tempo_bpm"),
        },
        "stems": {},
    }

    weighted_components: list[tuple[float, float]] = []
    for stem_name in DEFAULT_COMPARE_STEMS:
        stem_a = track_a["stems"][stem_name]
        stem_b = track_b["stems"][stem_name]
        source_events_a = stem_a.get("orchestral_note_events", stem_a["note_events"]) if stem_name == "other" else stem_a["note_events"]
        source_events_b = stem_b.get("orchestral_note_events", stem_b["note_events"]) if stem_name == "other" else stem_b["note_events"]
        events_a = _filter_events_to_window(source_events_a, resolved_window_a[0], resolved_window_a[1])
        events_b = _filter_events_to_window(source_events_b, resolved_window_b[0], resolved_window_b[1])

        fixed_sequence_a = _build_frame_sequence(
            events_a,
            window_duration_a,
            hop_sec=hop_sec,
            stem_name=stem_name,
            min_confidence=min_confidence,
            min_duration_sec=min_duration_sec,
        )
        fixed_sequence_b = _build_frame_sequence(
            events_b,
            window_duration_b,
            hop_sec=hop_sec,
            stem_name=stem_name,
            min_confidence=min_confidence,
            min_duration_sec=min_duration_sec,
        )

        best_fixed_pitch_class_similarity = -1.0
        best_fixed_shift = 0
        for shift in range(12):
            similarity = _dtw_similarity(fixed_sequence_a, _shift_pitch_classes(fixed_sequence_b, shift), _pitch_class_cost)
            if similarity > best_fixed_pitch_class_similarity:
                best_fixed_pitch_class_similarity = similarity
                best_fixed_shift = shift

        fixed_note_name_similarity = _dtw_similarity(fixed_sequence_a, fixed_sequence_b, _note_name_cost)
        fixed_combined_similarity = round(0.7 * best_fixed_pitch_class_similarity + 0.3 * fixed_note_name_similarity, 4)

        beat_boundaries_a = _window_beat_boundaries(
            beat_grid_a.get("beat_times", []),
            resolved_window_a[0],
            resolved_window_a[1],
            fallback_hop=hop_sec,
        )
        beat_boundaries_b = _window_beat_boundaries(
            beat_grid_b.get("beat_times", []),
            resolved_window_b[0],
            resolved_window_b[1],
            fallback_hop=hop_sec,
        )

        pitch_class_vectors_a, pitch_class_densities_a = _build_beat_vector_sequence(
            events_a,
            beat_boundaries_a,
            vector_mode="pitch_class",
            min_confidence=min_confidence,
            min_duration_sec=min_duration_sec,
        )
        pitch_class_vectors_b, pitch_class_densities_b = _build_beat_vector_sequence(
            events_b,
            beat_boundaries_b,
            vector_mode="pitch_class",
            min_confidence=min_confidence,
            min_duration_sec=min_duration_sec,
        )
        midi_vectors_a, _ = _build_beat_vector_sequence(
            events_a,
            beat_boundaries_a,
            vector_mode="midi",
            min_confidence=min_confidence,
            min_duration_sec=min_duration_sec,
        )
        midi_vectors_b, _ = _build_beat_vector_sequence(
            events_b,
            beat_boundaries_b,
            vector_mode="midi",
            min_confidence=min_confidence,
            min_duration_sec=min_duration_sec,
        )

        best_beat_pitch_similarity = -1.0
        best_beat_shift = 0
        for shift in range(12):
            similarity = _sequence_similarity(
                pitch_class_vectors_a,
                _shift_pitch_class_vectors(pitch_class_vectors_b, shift),
                _vector_cosine_cost,
            )
            if similarity > best_beat_pitch_similarity:
                best_beat_pitch_similarity = similarity
                best_beat_shift = shift

        beat_note_profile_similarity = _sequence_similarity(midi_vectors_a, midi_vectors_b, _vector_cosine_cost)

        contour_pitch_classes_a = _compress_pitch_classes(_dominant_pitch_classes(pitch_class_vectors_a, pitch_class_densities_a))
        contour_pitch_classes_b = _compress_pitch_classes(_dominant_pitch_classes(pitch_class_vectors_b, pitch_class_densities_b))
        interval_sequence_a = _interval_sequence(contour_pitch_classes_a)
        interval_sequence_b = _interval_sequence(contour_pitch_classes_b)
        contour_similarity = _sequence_similarity(interval_sequence_a, interval_sequence_b, _interval_cost)

        melodic_events_a = _extract_melodic_event_chain(
            events_a,
            stem_name=stem_name,
            min_confidence=min_confidence,
            min_duration_sec=min_duration_sec,
        )
        melodic_events_b = _extract_melodic_event_chain(
            events_b,
            stem_name=stem_name,
            min_confidence=min_confidence,
            min_duration_sec=min_duration_sec,
        )
        melodic_pitch_classes_a = _compress_pitch_classes(_event_chain_to_pitch_classes(melodic_events_a))
        melodic_pitch_classes_b = _compress_pitch_classes(_event_chain_to_pitch_classes(melodic_events_b))
        melodic_note_names_a = _event_chain_to_note_names(melodic_events_a)
        melodic_note_names_b = _event_chain_to_note_names(melodic_events_b)
        melodic_interval_sequence_a = _interval_sequence(melodic_pitch_classes_a)
        melodic_interval_sequence_b = _interval_sequence(melodic_pitch_classes_b)
        melodic_layer_similarity = _sequence_similarity(melodic_pitch_classes_a, melodic_pitch_classes_b, _pitch_class_cost)
        melodic_phrases_a = _segment_melodic_phrases(melodic_events_a)
        melodic_phrases_b = _segment_melodic_phrases(melodic_events_b)
        phrase_alignment = _build_phrase_alignment(melodic_phrases_a, melodic_phrases_b)
        motif_alignment = _build_motif_alignment(
            melodic_interval_sequence_a,
            melodic_interval_sequence_b,
            _note_classes_from_pitch_classes(melodic_pitch_classes_a),
            _note_classes_from_pitch_classes(melodic_pitch_classes_b),
        )

        beat_sync_combined_similarity = round(0.65 * best_beat_pitch_similarity + 0.35 * beat_note_profile_similarity, 4)
        if stem_name == "other":
            structural_similarity = round(
                0.45 * beat_sync_combined_similarity
                + 0.2 * contour_similarity
                + 0.15 * melodic_layer_similarity
                + 0.1 * float(motif_alignment["similarity"])
                + 0.1 * float(phrase_alignment["similarity"]),
                4,
            )
        else:
            structural_similarity = round(0.75 * beat_sync_combined_similarity + 0.25 * contour_similarity, 4)

        stem_weight = 0.7 if stem_name == "other" else 0.3
        weighted_components.append((stem_weight, structural_similarity))

        top_a = _top_notes(events_a, limit=8)
        top_b = _top_notes(events_b, limit=8)
        result["stems"][stem_name] = {
            "event_count_a": len(events_a),
            "event_count_b": len(events_b),
            "melodic_source_mode": "other_orchestral_proxy" if stem_name == "other" else "stem_note_events",
            "frame_count_a": len(fixed_sequence_a),
            "frame_count_b": len(fixed_sequence_b),
            "beat_count_a": max(0, len(beat_boundaries_a) - 1),
            "beat_count_b": max(0, len(beat_boundaries_b) - 1),
            "pitch_class_dtw_similarity": round(best_fixed_pitch_class_similarity, 4),
            "note_name_dtw_similarity": round(fixed_note_name_similarity, 4),
            "combined_similarity": fixed_combined_similarity,
            "beat_sync_pitch_class_similarity": round(best_beat_pitch_similarity, 4),
            "beat_sync_note_profile_similarity": round(beat_note_profile_similarity, 4),
            "beat_sync_combined_similarity": beat_sync_combined_similarity,
            "contour_similarity": round(contour_similarity, 4),
            "melodic_layer_similarity": round(melodic_layer_similarity, 4),
            "structural_similarity": structural_similarity,
            "best_transposition_semitones": best_beat_shift % 12,
            "legacy_best_transposition_semitones": best_fixed_shift % 12,
            "top_note_classes_a": top_a["top_note_classes"],
            "top_note_classes_b": top_b["top_note_classes"],
            "top_note_names_a": top_a["top_note_names"],
            "top_note_names_b": top_b["top_note_names"],
            "shared_top_note_classes": sorted(set(top_a["top_note_classes"]) & set(top_b["top_note_classes"])),
            "shared_top_note_names": sorted(set(top_a["top_note_names"]) & set(top_b["top_note_names"])),
            "contour_pitch_classes_a": contour_pitch_classes_a,
            "contour_pitch_classes_b": contour_pitch_classes_b,
            "contour_note_classes_a": _note_classes_from_pitch_classes(contour_pitch_classes_a),
            "contour_note_classes_b": _note_classes_from_pitch_classes(contour_pitch_classes_b),
            "interval_sequence_a": interval_sequence_a,
            "interval_sequence_b": interval_sequence_b,
            "melodic_layer_note_names_a": melodic_note_names_a,
            "melodic_layer_note_names_b": melodic_note_names_b,
            "melodic_layer_note_classes_a": _note_classes_from_pitch_classes(melodic_pitch_classes_a),
            "melodic_layer_note_classes_b": _note_classes_from_pitch_classes(melodic_pitch_classes_b),
            "melodic_layer_interval_sequence_a": melodic_interval_sequence_a,
            "melodic_layer_interval_sequence_b": melodic_interval_sequence_b,
            "melodic_event_chain_a": melodic_events_a,
            "melodic_event_chain_b": melodic_events_b,
            "melodic_phrases_a": melodic_phrases_a,
            "melodic_phrases_b": melodic_phrases_b,
            "motif_alignment": motif_alignment,
            "phrase_alignment": phrase_alignment,
        }

    total_weight = sum(weight for weight, _ in weighted_components) or 1.0
    overall_similarity = sum(weight * score for weight, score in weighted_components) / total_weight
    result["overall_melodic_similarity"] = round(float(overall_similarity), 4)
    return result
