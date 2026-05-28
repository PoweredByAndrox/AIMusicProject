from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import librosa
import numpy as np
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
import essentia.standard as essentia_std
from .device_policy import resolve_torch_device

NOTE_NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
SEPARATION_SAMPLE_RATE = 44100
FEATURE_SAMPLE_RATE = 22050
PITCH_HOP_SIZE = 128
CHROMA_HOP_LENGTH = 512
_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _emit_progress(message: str) -> None:
    print(f'[instrument_melody_profile] {message}', flush=True)


def _resolve_device(device: str | None = None) -> str:
    return str(resolve_torch_device(task_name='instrument_melody_profile', requested=device, allow_cpu=False))


def _load_demucs_model(model_name: str, device: str):
    cache_key = (model_name, device)
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        model = get_model(model_name)
        if hasattr(model, 'to'):
            model = model.to(device)
        _MODEL_CACHE[cache_key] = model
    return model


def _ensure_stereo(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError(f'Expected [channels, time] waveform, got shape={tuple(waveform.shape)}')
    if waveform.shape[0] == 1:
        return waveform.repeat(2, 1)
    if waveform.shape[0] >= 2:
        return waveform[:2, :]
    raise ValueError('Waveform has no channels')


def separate_track_stems(
    audio_path: str | Path,
    *,
    model_name: str = 'htdemucs',
    device: str | None = None,
) -> dict[str, np.ndarray]:
    _emit_progress(f'ENTER separate_track_stems model={model_name} audio_path={audio_path}')
    resolved_device = _resolve_device(device)
    model = _load_demucs_model(model_name, resolved_device)
    waveform, sample_rate = torchaudio.load(str(audio_path))
    waveform = _ensure_stereo(waveform)
    if sample_rate != SEPARATION_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SEPARATION_SAMPLE_RATE)

    with torch.no_grad():
        separated = apply_model(
            model,
            waveform.unsqueeze(0),
            split=True,
            progress=True,
            device=resolved_device,
        )

    if separated.ndim != 4 or separated.shape[0] != 1:
        raise ValueError(f'Unexpected Demucs output shape: {tuple(separated.shape)}')

    stems: dict[str, np.ndarray] = {}
    for index, stem_name in enumerate(model.sources):
        stem_audio = separated[0, index].detach().cpu().mean(dim=0).numpy().astype(np.float32)
        if SEPARATION_SAMPLE_RATE != FEATURE_SAMPLE_RATE:
            stem_audio = librosa.resample(
                stem_audio,
                orig_sr=SEPARATION_SAMPLE_RATE,
                target_sr=FEATURE_SAMPLE_RATE,
            ).astype(np.float32)
        stems[str(stem_name)] = stem_audio

    if resolved_device == 'cuda':
        torch.cuda.empty_cache()

    _emit_progress(f'EXIT separate_track_stems model={model_name} stems={list(stems.keys())} sample_rate={FEATURE_SAMPLE_RATE}')
    return stems


def _slice_audio(audio: np.ndarray, start_time: float, end_time: float, sample_rate: int) -> np.ndarray:
    start_sample = max(0, int(round(start_time * sample_rate)))
    end_sample = max(start_sample, int(round(end_time * sample_rate)))
    return audio[start_sample:end_sample].astype(np.float32, copy=False)


def _top_chroma_notes(audio: np.ndarray, sample_rate: int, limit: int = 4) -> tuple[list[str], list[float]]:
    if audio.size == 0:
        return [], []

    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=sample_rate,
        hop_length=CHROMA_HOP_LENGTH,
        n_chroma=12,
    )
    if chroma.size == 0:
        return [], []

    avg = chroma.mean(axis=1)
    top_indices = avg.argsort()[::-1][:limit]
    return (
        [NOTE_NAMES[int(index)] for index in top_indices],
        [round(float(avg[int(index)]), 4) for index in top_indices],
    )


def _pitch_summary(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    if audio.size < sample_rate:
        return {
            'melodic_ratio': 0.0,
            'median_midi': None,
            'pitch_top_notes': [],
            'mean_confidence': 0.0,
        }

    extractor = essentia_std.PredominantPitchMelodia(
        frameSize=2048,
        hopSize=PITCH_HOP_SIZE,
        sampleRate=sample_rate,
        minFrequency=40,
        maxFrequency=2200,
    )
    pitch, confidence = extractor(audio)
    active = (pitch > 0) & (confidence >= 0.35)
    if not np.any(active):
        return {
            'melodic_ratio': 0.0,
            'median_midi': None,
            'pitch_top_notes': [],
            'mean_confidence': 0.0,
        }

    midi = 69 + 12 * np.log2(pitch[active] / 440.0)
    rounded = np.round(midi).astype(int)
    pitch_classes = rounded % 12
    counts = np.bincount(pitch_classes, minlength=12)
    top_indices = counts.argsort()[::-1][:3]

    return {
        'melodic_ratio': round(float(active.mean()), 4),
        'median_midi': round(float(np.median(midi)), 2),
        'pitch_top_notes': [NOTE_NAMES[int(index)] for index in top_indices if counts[int(index)] > 0],
        'mean_confidence': round(float(confidence[active].mean()), 4),
    }


def build_track_instrument_melody_profile(
    audio_path: str | Path,
    segments: Sequence[Mapping[str, Any]],
    *,
    model_name: str = 'htdemucs',
    device: str | None = None,
) -> list[dict[str, Any]]:
    stems = separate_track_stems(audio_path, model_name=model_name, device=device)
    profiles: list[dict[str, Any]] = []

    for segment in segments:
        start_time = float(segment['start'])
        end_time = float(segment['end'])
        segment_result: dict[str, Any] = {
            'segment_id': int(segment['segment_id']),
            'type': str(segment['type']),
            'start': round(start_time, 3),
            'end': round(end_time, 3),
            'stems': {},
        }

        raw_features: Dict[str, dict[str, Any]] = {}
        total_energy = 0.0
        for stem_name, stem_audio in stems.items():
            clip = _slice_audio(stem_audio, start_time, end_time, FEATURE_SAMPLE_RATE)
            energy = float(np.sqrt(np.mean(clip ** 2))) if clip.size else 0.0
            total_energy += energy
            chroma_notes, chroma_values = _top_chroma_notes(clip, FEATURE_SAMPLE_RATE)
            pitch_features = _pitch_summary(clip, FEATURE_SAMPLE_RATE)
            raw_features[stem_name] = {
                'energy': energy,
                'top_chroma_notes': chroma_notes,
                'top_chroma_values': chroma_values,
                **pitch_features,
            }

        normalized_energy = total_energy or 1.0
        for stem_name, features in raw_features.items():
            segment_result['stems'][stem_name] = {
                'energy_ratio': round(float(features['energy'] / normalized_energy), 4),
                'melodic_ratio': features['melodic_ratio'],
                'median_midi': features['median_midi'],
                'mean_confidence': features['mean_confidence'],
                'pitch_top_notes': features['pitch_top_notes'],
                'top_chroma_notes': features['top_chroma_notes'],
                'top_chroma_values': features['top_chroma_values'],
            }

        profiles.append(segment_result)

    return profiles
