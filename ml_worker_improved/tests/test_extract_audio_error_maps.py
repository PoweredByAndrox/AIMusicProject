import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.extract_audio_error_maps import compute_audio_error_maps, extract_error_maps_for_window, write_error_maps


SAMPLE_RATE = 8_000


def sine(freq: float, seconds: float, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def stereo(audio: np.ndarray) -> np.ndarray:
    return np.column_stack([audio, audio * 0.85]).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return str(path)


def test_compute_audio_error_maps_writes_time_frequency_arrays(tmp_path: Path) -> None:
    vocal = sine(440, 1.0, amp=0.20)
    minus = sine(110, 1.0, amp=0.30)
    artifact = sine(1_600, 1.0, amp=0.04)
    mix = vocal + minus + artifact

    result = compute_audio_error_maps(
        stereo(mix),
        stereo(vocal),
        stereo(minus),
        sample_rate=SAMPLE_RATE,
        map_max_bins=48,
        map_max_frames=32,
        max_waveform_samples=256,
    )

    assert result["metrics"]["residual_energy"] > 0.0
    assert result["metrics"]["mixture_consistency_error"] > 0.0
    assert result["metrics"]["artifact_noise_score"] > 0.0
    assert result["release_ready"] is False
    assert result["melody_transcription_ran"] is False
    assert result["recommendation_index_updated"] is False

    npz_path = write_error_maps(
        output_dir=tmp_path / "maps",
        dataset_id="dataset-test",
        row_id="track-a__cand-a__vocal__0000",
        maps=result["maps"],
        metadata=result["metadata"],
    )
    assert npz_path.exists()

    arrays = np.load(npz_path)
    for key in (
        "waveform_residual",
        "stft_residual_magnitude",
        "phase_difference_map",
        "stereo_width_map",
        "vocal_bleed_proxy_map",
        "instrumental_bleed_proxy_map",
        "missing_detail_proxy_map",
        "transient_loss_proxy_map",
        "artifact_noise_proxy_map",
        "reconstruction_error_map",
    ):
        assert key in arrays.files
        assert arrays[key].size > 0
        assert np.isfinite(arrays[key]).all()


def test_extract_error_maps_for_window_reads_bounded_audio(tmp_path: Path) -> None:
    vocal = sine(330, 2.0, amp=0.16)
    minus = sine(90, 2.0, amp=0.22)
    mix = vocal + minus
    mix_path = write_wav(tmp_path / "mix.wav", stereo(mix))
    vocal_path = write_wav(tmp_path / "vocal.wav", stereo(vocal))
    minus_path = write_wav(tmp_path / "minus.wav", stereo(minus))

    result = extract_error_maps_for_window(
        mix_path=Path(mix_path),
        vocal_path=Path(vocal_path),
        minus_path=Path(minus_path),
        window_start_sec=0.25,
        window_end_sec=0.75,
        map_max_bins=32,
        map_max_frames=16,
        max_waveform_samples=128,
    )

    assert result["sample_rate"] == SAMPLE_RATE
    assert result["channels"] == 2
    assert 0.49 <= result["duration_sec"] <= 0.51
    assert result["metrics"]["mixture_consistency_error"] < 0.01
