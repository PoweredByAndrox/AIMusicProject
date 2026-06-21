"""Shared feature extraction and torch model for stem contamination checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.stem_contamination_common import FORBIDDEN_SOURCES, MINUS_STEM, STEMS


LEGACY_BASE_FEATURE_NAMES = (
    "vocal_band_dbfs",
    "low_band_dbfs",
    "mid_band_dbfs",
    "high_band_dbfs",
    "total_dbfs",
    "vocal_to_low_db",
    "vocal_to_total_db",
)
BASE_FEATURE_NAMES = (
    *LEGACY_BASE_FEATURE_NAMES,
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff85_hz",
    "spectral_flatness",
    "zero_crossing_rate",
    "crest_factor_db",
    "envelope_cv",
    "transient_ratio",
)
PAIR_FEATURE_NAMES = (
    "foreign_band_dbfs",
    "foreign_to_total_db",
    "foreign_to_vocal_db",
    "foreign_to_low_db",
    "foreign_to_high_db",
)
TESTED_STEM_FEATURES = (*STEMS, MINUS_STEM)
FOREIGN_SOURCE_FEATURES = STEMS
LEGACY_FEATURE_NAMES = (
    *LEGACY_BASE_FEATURE_NAMES,
    *(f"tested_stem={stem}" for stem in TESTED_STEM_FEATURES),
    *(f"foreign_source={stem}" for stem in FOREIGN_SOURCE_FEATURES),
)
FEATURE_NAMES = (
    *BASE_FEATURE_NAMES,
    *PAIR_FEATURE_NAMES,
    *(f"tested_stem={stem}" for stem in TESTED_STEM_FEATURES),
    *(f"foreign_source={stem}" for stem in FOREIGN_SOURCE_FEATURES),
)
VOCAL_BAND_HZ = (700.0, 1200.0)
SOURCE_BANDS_HZ = {
    "vocals": VOCAL_BAND_HZ,
    "drums": (1200.0, 6000.0),
    "bass": (40.0, 300.0),
    "guitar": (300.0, 1600.0),
    "piano": (250.0, 4000.0),
    "other": (300.0, 4000.0),
}

try:  # pragma: no cover - exercised in Docker tests.
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def dbfs(value: float) -> float:
    if value <= 1e-12:
        return -240.0
    return 20.0 * math.log10(value)


def band_limited_audio(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    if len(audio) <= 0:
        return np.zeros_like(audio, dtype=np.float32)
    working = audio.astype(np.float64)
    spectrum = np.fft.rfft(working, axis=0)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / float(sample_rate))
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    filtered = np.zeros_like(spectrum)
    filtered[mask, :] = spectrum[mask, :]
    return np.fft.irfft(filtered, n=len(audio), axis=0).astype(np.float32)


def _one_hot(value: str, choices: tuple[str, ...]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def source_band_limited_audio(audio: np.ndarray, sample_rate: int, foreign_source: str) -> np.ndarray:
    low_hz, high_hz = SOURCE_BANDS_HZ.get(foreign_source, VOCAL_BAND_HZ)
    return band_limited_audio(audio, sample_rate, low_hz, high_hz)


def _mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array
    return np.mean(array, axis=1).astype(np.float32)


def _safe_ratio_db(numerator: float, denominator: float) -> float:
    return dbfs(float(numerator)) - dbfs(float(denominator))


def _spectral_shape_features(audio: np.ndarray, sample_rate: int) -> list[float]:
    mono = _mono(audio)
    if mono.size <= 1:
        return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    window = np.hanning(len(mono)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(mono * window)).astype(np.float64)
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / float(sample_rate))
    spectrum = np.maximum(spectrum, 1.0e-12)
    magnitude_sum = float(np.sum(spectrum))
    centroid = float(np.sum(freqs * spectrum) / magnitude_sum)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / magnitude_sum))
    cumulative = np.cumsum(spectrum)
    rolloff_index = int(np.searchsorted(cumulative, cumulative[-1] * 0.85))
    rolloff = float(freqs[min(rolloff_index, len(freqs) - 1)])
    flatness = float(np.exp(np.mean(np.log(spectrum))) / (np.mean(spectrum) + 1.0e-12))
    zero_crossings = float(np.mean(np.abs(np.diff(np.signbit(mono)).astype(np.float32))))
    audio_rms = rms(mono)
    crest_db = dbfs(float(np.max(np.abs(mono)))) - dbfs(audio_rms)

    frame = max(128, int(sample_rate * 0.02))
    hop = max(1, frame // 2)
    if len(mono) < frame:
        envelope = np.asarray([audio_rms], dtype=np.float32)
    else:
        envelope = np.asarray(
            [
                rms(mono[start : start + frame])
                for start in range(0, len(mono) - frame + 1, hop)
            ],
            dtype=np.float32,
        )
    envelope_mean = float(np.mean(envelope) + 1.0e-12)
    envelope_cv = float(np.std(envelope) / envelope_mean)
    transient_ratio = float(np.mean(np.abs(np.diff(envelope))) / envelope_mean) if len(envelope) > 1 else 0.0
    return [
        centroid,
        bandwidth,
        rolloff,
        flatness,
        zero_crossings,
        crest_db,
        envelope_cv,
        transient_ratio,
    ]


def _legacy_extract_base_no_reference_features(audio: np.ndarray, sample_rate: int) -> list[float]:
    vocal_db = dbfs(rms(band_limited_audio(audio, sample_rate, 700.0, 1200.0)))
    low_db = dbfs(rms(band_limited_audio(audio, sample_rate, 80.0, 300.0)))
    mid_db = dbfs(rms(band_limited_audio(audio, sample_rate, 300.0, 700.0)))
    high_db = dbfs(rms(band_limited_audio(audio, sample_rate, 1200.0, 4000.0)))
    total_db = dbfs(rms(audio))
    return [
        vocal_db,
        low_db,
        mid_db,
        high_db,
        total_db,
        vocal_db - low_db,
        vocal_db - total_db,
    ]


def extract_base_no_reference_features(audio: np.ndarray, sample_rate: int) -> list[float]:
    legacy = _legacy_extract_base_no_reference_features(audio, sample_rate)
    return [
        *legacy,
        *_spectral_shape_features(audio, sample_rate),
    ]


def extract_pair_no_reference_features(
    audio: np.ndarray,
    sample_rate: int,
    *,
    foreign_source: str = "",
) -> list[float]:
    vocal_rms = rms(band_limited_audio(audio, sample_rate, 700.0, 1200.0))
    low_rms = rms(band_limited_audio(audio, sample_rate, 80.0, 300.0))
    high_rms = rms(band_limited_audio(audio, sample_rate, 1200.0, 4000.0))
    total_rms = rms(audio)
    foreign_rms = rms(source_band_limited_audio(audio, sample_rate, foreign_source))
    return [
        dbfs(foreign_rms),
        _safe_ratio_db(foreign_rms, total_rms),
        _safe_ratio_db(foreign_rms, vocal_rms),
        _safe_ratio_db(foreign_rms, low_rms),
        _safe_ratio_db(foreign_rms, high_rms),
    ]


def _legacy_extract_no_reference_features(
    audio: np.ndarray,
    sample_rate: int,
    *,
    tested_stem: str = "",
    foreign_source: str = "",
) -> list[float]:
    return [
        *_legacy_extract_base_no_reference_features(audio, sample_rate),
        *_one_hot(tested_stem, TESTED_STEM_FEATURES),
        *_one_hot(foreign_source, FOREIGN_SOURCE_FEATURES),
    ]


def extract_no_reference_features(
    audio: np.ndarray,
    sample_rate: int,
    *,
    tested_stem: str = "",
    foreign_source: str = "",
) -> list[float]:
    base_features = extract_base_no_reference_features(audio, sample_rate)
    return [
        *base_features,
        *extract_pair_no_reference_features(audio, sample_rate, foreign_source=foreign_source),
        *_one_hot(tested_stem, TESTED_STEM_FEATURES),
        *_one_hot(foreign_source, FOREIGN_SOURCE_FEATURES),
    ]


def _trained_pair_set(manifest: dict[str, Any]) -> set[tuple[str, str]]:
    trained_pairs = manifest.get("trained_pairs")
    if isinstance(trained_pairs, list):
        pairs: set[tuple[str, str]] = set()
        for item in trained_pairs:
            if not isinstance(item, dict):
                continue
            tested_stem = str(item.get("tested_stem") or "")
            foreign_source = str(item.get("foreign_source") or "")
            if tested_stem and foreign_source:
                pairs.add((tested_stem, foreign_source))
        return pairs
    tested_stem = str(manifest.get("tested_stem") or "")
    foreign_source = str(manifest.get("foreign_source") or "")
    return {(tested_stem, foreign_source)} if tested_stem and foreign_source else set()


if TORCH_AVAILABLE:

    class StemContaminationNet(nn.Module):
        def __init__(self, input_width: int, hidden_size: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_width, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, features: Any) -> Any:
            return torch.sigmoid(self.net(features))

else:

    class StemContaminationNet:  # pragma: no cover - only used when torch import fails.
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("torch_unavailable")


def predict_no_reference_probability(
    *,
    model_manifest_path: Path,
    audio: np.ndarray,
    sample_rate: int,
    tested_stem: str,
    foreign_source: str,
) -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch_unavailable")
    manifest = json.loads(model_manifest_path.read_text(encoding="utf-8-sig"))
    if foreign_source not in FORBIDDEN_SOURCES.get(tested_stem, ()):
        raise ValueError(f"invalid_forbidden_pair:{tested_stem}:{foreign_source}")
    if (tested_stem, foreign_source) not in _trained_pair_set(manifest):
        raise ValueError(f"model_pair_untrained:{tested_stem}:{foreign_source}")
    model_file = Path(str(manifest.get("model_file") or ""))
    if not model_file.is_file():
        raise FileNotFoundError(f"missing model file: {model_file}")
    try:
        bundle = torch.load(model_file, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch compatibility.
        bundle = torch.load(model_file, map_location="cpu")
    input_width = int(bundle["input_width"])
    hidden_size = int(bundle["hidden_size"])
    model = StemContaminationNet(input_width=input_width, hidden_size=hidden_size)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    bundle_feature_names = list(bundle.get("feature_names") or [])
    if bundle_feature_names == list(LEGACY_BASE_FEATURE_NAMES) or input_width == len(LEGACY_BASE_FEATURE_NAMES):
        features = np.asarray(_legacy_extract_base_no_reference_features(audio, sample_rate), dtype=np.float32)
    elif bundle_feature_names == list(LEGACY_FEATURE_NAMES) or input_width == len(LEGACY_FEATURE_NAMES):
        features = np.asarray(
            _legacy_extract_no_reference_features(audio, sample_rate, tested_stem=tested_stem, foreign_source=foreign_source),
            dtype=np.float32,
        )
    elif bundle_feature_names == list(BASE_FEATURE_NAMES) or input_width == len(BASE_FEATURE_NAMES):
        features = np.asarray(extract_base_no_reference_features(audio, sample_rate), dtype=np.float32)
    elif bundle_feature_names == list(FEATURE_NAMES) or input_width == len(FEATURE_NAMES):
        features = np.asarray(
            extract_no_reference_features(audio, sample_rate, tested_stem=tested_stem, foreign_source=foreign_source),
            dtype=np.float32,
        )
    else:
        raise ValueError(f"unsupported feature vector width: {input_width}")
    mean = np.asarray(bundle["feature_mean"], dtype=np.float32)
    std = np.asarray(bundle["feature_std"], dtype=np.float32)
    normalized = (features - mean) / np.maximum(std, 1.0e-6)
    with torch.no_grad():
        probability = float(model(torch.tensor(normalized[None, :], dtype=torch.float32)).cpu().item())
    return {
        "probability": probability,
        "threshold": float(manifest.get("threshold", 0.5)),
        "model_id": str(manifest.get("model_id") or ""),
        "model_manifest": str(model_manifest_path),
        "feature_names": list(bundle.get("feature_names") or FEATURE_NAMES),
        "features": [round(float(value), 6) for value in features.tolist()],
        "tested_stem": tested_stem,
        "foreign_source": foreign_source,
    }
