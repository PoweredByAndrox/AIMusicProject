from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.build_procedural_stem_clean_pairs import build_procedural_stem_clean_pairs
from tools.build_stem_contamination_dataset import build_dataset_from_manifest, validate_stem_contamination_row
import tools.score_stem_contamination as scorer
from tools.score_stem_contamination import score_stem_contamination
from tools.stem_contamination_common import build_minus_from_non_vocal_stems
from tools.stem_contamination_model import BASE_FEATURE_NAMES, StemContaminationNet
from tools.train_stem_contamination_detector import train_stem_contamination_detector


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float = 0.25, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def voice_like_vocal(seconds: float = 0.75) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    f0 = 185.0 + 18.0 * np.sin(2.0 * math.pi * 4.5 * t)
    phase = np.cumsum(2.0 * math.pi * f0 / float(SAMPLE_RATE)).astype(np.float32)
    audio = np.zeros_like(t)
    for harmonic, gain in ((1, 0.13), (2, 0.09), (3, 0.07), (4, 0.045), (5, 0.035), (7, 0.026), (9, 0.018)):
        audio += gain * np.sin(float(harmonic) * phase + 0.17 * float(harmonic))
    envelope = 0.52 + 0.48 * np.sin(2.0 * math.pi * 2.2 * t) ** 2
    return (audio * envelope).astype(np.float32)


def bright_voice_like_vocal(seconds: float = 0.75) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    f0 = 155.0 + 24.0 * np.sin(2.0 * math.pi * 5.3 * t)
    phase = np.cumsum(2.0 * math.pi * f0 / float(SAMPLE_RATE)).astype(np.float32)
    audio = np.zeros_like(t)
    for harmonic in range(1, 18):
        tilt = 0.12 / float(harmonic) ** 0.82
        formant = 1.0
        harmonic_hz = float(harmonic) * 180.0
        if 650.0 <= harmonic_hz <= 1100.0:
            formant *= 1.55
        if 1800.0 <= harmonic_hz <= 3200.0:
            formant *= 0.75
        audio += tilt * formant * np.sin(float(harmonic) * phase + 0.11 * harmonic).astype(np.float32)
    rng = np.random.default_rng(321)
    fricative = rng.normal(0.0, 0.012, len(t)).astype(np.float32)
    fricative *= (np.sin(2.0 * math.pi * 6.0 * t) > 0.45).astype(np.float32)
    envelope = 0.46 + 0.54 * np.sin(2.0 * math.pi * 2.7 * t) ** 2
    return ((audio + fricative) * envelope).astype(np.float32)


def kick_drum_like(seconds: float = 0.75) -> np.ndarray:
    frames = int(SAMPLE_RATE * seconds)
    audio = np.zeros(frames, dtype=np.float32)
    pulse_spacing = int(SAMPLE_RATE * 0.18)
    decay_len = int(SAMPLE_RATE * 0.09)
    t = np.arange(decay_len, dtype=np.float32) / float(SAMPLE_RATE)
    kick = 0.34 * np.sin(2.0 * math.pi * 82.0 * t) * np.exp(-t * 38.0)
    click = np.zeros(decay_len, dtype=np.float32)
    click[: max(1, int(SAMPLE_RATE * 0.004))] = 0.22
    for start in range(0, frames, pulse_spacing):
        end = min(frames, start + decay_len)
        audio[start:end] += (kick[: end - start] + click[: end - start]).astype(np.float32)
    return audio.astype(np.float32)


def boomy_kick_drum_like(seconds: float = 0.75) -> np.ndarray:
    frames = int(SAMPLE_RATE * seconds)
    audio = np.zeros(frames, dtype=np.float32)
    pulse_spacing = int(SAMPLE_RATE * 0.19)
    decay_len = int(SAMPLE_RATE * 0.16)
    t = np.arange(decay_len, dtype=np.float32) / float(SAMPLE_RATE)
    pitch = 92.0 - 36.0 * np.minimum(t / 0.11, 1.0)
    phase = np.cumsum(2.0 * math.pi * pitch / float(SAMPLE_RATE)).astype(np.float32)
    kick = 0.42 * np.sin(phase) * np.exp(-t * 18.0)
    click = np.zeros(decay_len, dtype=np.float32)
    click[: max(1, int(SAMPLE_RATE * 0.003))] = 0.24
    for start in range(0, frames, pulse_spacing):
        end = min(frames, start + decay_len)
        audio[start:end] += (kick[: end - start] + click[: end - start]).astype(np.float32)
    return audio.astype(np.float32)


def sustained_bass_tone(seconds: float = 0.75) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    return (0.16 * np.sin(2.0 * math.pi * 74.0 * t)).astype(np.float32)


def harmonic_bass_like(seconds: float = 0.75) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    envelope = 0.86 + 0.14 * np.sin(2.0 * math.pi * 3.1 * t)
    audio = np.zeros_like(t)
    for harmonic, gain in ((1, 0.20), (2, 0.12), (3, 0.075), (4, 0.045), (5, 0.027), (6, 0.016)):
        audio += gain * np.sin(2.0 * math.pi * 61.0 * float(harmonic) * t + 0.07 * harmonic).astype(np.float32)
    return (audio * envelope).astype(np.float32)


def guitar_like_stem(seconds: float = 0.75) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    envelope = np.exp(-t * 1.8).astype(np.float32) * (0.82 + 0.18 * np.sin(2.0 * math.pi * 5.0 * t))
    audio = np.zeros_like(t)
    for freq, gain in ((196.0, 0.12), (392.0, 0.08), (588.0, 0.05), (784.0, 0.032), (1176.0, 0.018)):
        audio += gain * np.sin(2.0 * math.pi * freq * t + 0.13 * gain).astype(np.float32)
    return (audio * envelope).astype(np.float32)


def cymbal_like_drums(seconds: float = 0.75) -> np.ndarray:
    frames = int(SAMPLE_RATE * seconds)
    audio = np.zeros(frames, dtype=np.float32)
    rng = np.random.default_rng(4242)
    pulse_spacing = int(SAMPLE_RATE * 0.16)
    decay_len = int(SAMPLE_RATE * 0.05)
    decay = np.exp(-np.arange(decay_len, dtype=np.float32) / float(SAMPLE_RATE * 0.008))
    for start in range(0, frames, pulse_spacing):
        end = min(frames, start + decay_len)
        noise = rng.normal(0.0, 0.25, end - start).astype(np.float32)
        audio[start:end] += noise * decay[: end - start]
    return audio.astype(np.float32)


def sustained_mid_tone(freq: float, seconds: float = 0.75, amp: float = 0.12) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    return (float(amp) * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return path


def read_wav(path: Path) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32)


def test_minus_is_sum_of_non_vocal_stems(tmp_path: Path) -> None:
    stems = {
        "drums": write_wav(tmp_path / "drums.wav", sine(110.0, amp=0.05)),
        "bass": write_wav(tmp_path / "bass.wav", sine(220.0, amp=0.04)),
        "guitar": write_wav(tmp_path / "guitar.wav", sine(330.0, amp=0.03)),
        "piano": write_wav(tmp_path / "piano.wav", sine(440.0, amp=0.02)),
        "other": write_wav(tmp_path / "other.wav", sine(550.0, amp=0.01)),
    }

    out = build_minus_from_non_vocal_stems(stems, tmp_path / "minus.wav")

    expected = sum(read_wav(path) for path in stems.values())
    np.testing.assert_allclose(read_wav(out), expected, atol=2e-4)


def test_generated_separator_output_cannot_be_ground_truth() -> None:
    with pytest.raises(ValueError, match="generated.*ground truth"):
        validate_stem_contamination_row(
            {
                "row_id": "d000001",
                "input_stem": "audio/in.flac",
                "target_leak": "audio/leak.flac",
                "base_stem": "guitar",
                "foreign_source": "vocals",
                "label_type": "generated_separator",
                "is_ground_truth": True,
            }
        )


def test_dirty_ref_cannot_be_exact_ground_truth() -> None:
    with pytest.raises(ValueError, match="dirty.*ground truth"):
        validate_stem_contamination_row(
            {
                "row_id": "d000002",
                "input_stem": "audio/in.flac",
                "target_leak": "audio/leak.flac",
                "base_stem": "minus",
                "foreign_source": "vocals",
                "label_type": "dirty_ref",
                "is_ground_truth": True,
            }
        )


def test_invalid_forbidden_pair_row_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid forbidden source pair"):
        validate_stem_contamination_row(
            {
                "row_id": "d000003",
                "input_stem": "audio/in.flac",
                "target_leak": "audio/leak.flac",
                "base_stem": "vocals",
                "foreign_source": "vocals",
                "label_type": "synthetic_clean",
                "is_ground_truth": True,
            }
        )


def test_vocal_injected_into_minus_is_reconstructed_as_leak(tmp_path: Path) -> None:
    base_minus = sine(110.0, amp=0.2)
    injected_vocal = sine(880.0, amp=0.03)
    dirty_minus = base_minus + injected_vocal

    stem_path = write_wav(tmp_path / "minus.wav", dirty_minus)
    vocal_ref = write_wav(tmp_path / "vocal_ref.wav", injected_vocal)

    result = score_stem_contamination(
        tested_stem_path=stem_path,
        tested_stem="minus",
        foreign_source="vocals",
        output_dir=tmp_path / "score",
        foreign_reference_path=vocal_ref,
        detection_floor_db=-45.0,
    )

    assert result["verdict"] == "fail"
    assert result["foreign_source"] == "vocals"
    assert result["leak_audio"]
    leak_audio = read_wav(tmp_path / "score" / result["leak_audio"])
    assert float(np.sqrt(np.mean(np.square(leak_audio)))) > 0.01


def test_clean_minus_does_not_emit_vocal_leak(tmp_path: Path) -> None:
    stem_path = write_wav(tmp_path / "minus.wav", sine(110.0, amp=0.2))
    vocal_ref = write_wav(tmp_path / "vocal_ref.wav", sine(880.0, amp=0.03))

    result = score_stem_contamination(
        tested_stem_path=stem_path,
        tested_stem="minus",
        foreign_source="vocals",
        output_dir=tmp_path / "score",
        foreign_reference_path=vocal_ref,
        detection_floor_db=-45.0,
    )

    assert result["verdict"] == "pass"
    assert result["leak_audio"] == ""
    assert json.loads((tmp_path / "score" / "minus_vocals.json").read_text(encoding="utf-8"))["verdict"] == "pass"


def test_no_reference_vocal_like_leak_in_minus_is_reconstructed(tmp_path: Path) -> None:
    base_minus = sine(110.0, amp=0.2)
    leaked_vocal = sine(880.0, amp=0.04)
    dirty_minus = base_minus + leaked_vocal

    result = score_stem_contamination(
        tested_stem_path=write_wav(tmp_path / "minus.wav", dirty_minus),
        tested_stem="minus",
        foreign_source="vocals",
        output_dir=tmp_path / "score",
        foreign_reference_path=None,
        detection_floor_db=-45.0,
    )

    assert result["verdict"] == "fail"
    assert result["detector_mode"] == "no_reference_spectral_v1"
    assert result["leak_audio"]
    assert result["regions"]
    leak_audio = read_wav(tmp_path / "score" / result["leak_audio"])
    assert float(np.sqrt(np.mean(np.square(leak_audio)))) > 0.01


def test_no_reference_clean_minus_passes_without_reference(tmp_path: Path) -> None:
    clean_minus = sine(110.0, amp=0.2) + sine(220.0, amp=0.04)

    result = score_stem_contamination(
        tested_stem_path=write_wav(tmp_path / "minus.wav", clean_minus),
        tested_stem="minus",
        foreign_source="vocals",
        output_dir=tmp_path / "score",
        foreign_reference_path=None,
        detection_floor_db=-45.0,
    )

    assert result["verdict"] == "pass"
    assert result["detector_mode"] == "no_reference_spectral_v1"
    assert result["leak_audio"] == ""


def test_legacy_single_pair_model_manifest_remains_scorable(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model = StemContaminationNet(input_width=len(BASE_FEATURE_NAMES), hidden_size=8)
    model_file = tmp_path / "legacy_model.pt"
    torch.save(
        {
            "model_id": "legacy_scd",
            "input_width": len(BASE_FEATURE_NAMES),
            "hidden_size": 8,
            "feature_names": list(BASE_FEATURE_NAMES),
            "feature_mean": [0.0 for _ in BASE_FEATURE_NAMES],
            "feature_std": [1.0 for _ in BASE_FEATURE_NAMES],
            "state_dict": model.state_dict(),
        },
        model_file,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "stem_contamination_detector_model.v1",
                "model_id": "legacy_scd",
                "model_file": str(model_file),
                "tested_stem": "minus",
                "foreign_source": "vocals",
                "threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )

    result = score_stem_contamination(
        tested_stem_path=write_wav(tmp_path / "minus.wav", sine(110.0, amp=0.2) + sine(880.0, amp=0.04)),
        tested_stem="minus",
        foreign_source="vocals",
        output_dir=tmp_path / "legacy_score",
        foreign_reference_path=None,
        model_manifest_path=manifest_path,
    )

    assert result["verdict"] in {"pass", "fail"}
    assert result["detector_mode"] == "no_reference_neural_v1"


def test_no_reference_model_probability_without_leak_audio_stays_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    def high_probability_without_acoustic_leak(**_kwargs: object) -> dict[str, object]:
        return {
            "probability": 0.99,
            "threshold": 0.5,
            "model_id": "fixture",
            "model_manifest": str(manifest_path),
            "features": [],
            "feature_names": [],
        }

    monkeypatch.setattr(scorer, "predict_no_reference_probability", high_probability_without_acoustic_leak)

    result = score_stem_contamination(
        tested_stem_path=write_wav(tmp_path / "guitar.wav", sine(110.0, amp=0.2)),
        tested_stem="guitar",
        foreign_source="vocals",
        output_dir=tmp_path / "score",
        foreign_reference_path=None,
        model_manifest_path=manifest_path,
        detection_floor_db=-45.0,
    )

    assert result["verdict"] == "pass"
    assert result["detector_mode"] == "no_reference_neural_v1"
    assert result["contamination_probability"] == 0.99
    assert result["leak_dbfs"] <= result["detection_floor_db"]
    assert result["leak_audio"] == ""


def test_no_reference_unsupported_pair_blocks_unknown(tmp_path: Path) -> None:
    result = score_stem_contamination(
        tested_stem_path=write_wav(tmp_path / "guitar.wav", sine(330.0, amp=0.2)),
        tested_stem="guitar",
        foreign_source="piano",
        output_dir=tmp_path / "score",
        foreign_reference_path=None,
        detection_floor_db=-45.0,
    )

    assert result["verdict"] == "blocked_unknown"
    assert result["reason"] == "no_reference_detector_not_available"


def _write_contamination_rows(tmp_path: Path) -> Path:
    rows = []
    for idx, freq in enumerate((820.0, 880.0, 960.0), start=1):
        clean_minus = sine(110.0 + idx * 20.0, amp=0.18) + sine(220.0, amp=0.04)
        leak = sine(freq, amp=0.045)
        dirty = clean_minus + leak
        row_id = f"d{idx:06d}"
        input_path = write_wav(tmp_path / "audio" / f"{row_id}_in.flac", dirty)
        leak_path = write_wav(tmp_path / "audio" / f"{row_id}_leak.flac", leak)
        rows.append(
            {
                "row_id": row_id,
                "input_stem": str(input_path),
                "target_leak": str(leak_path),
                "base_stem": "minus",
                "foreign_source": "vocals",
                "label_type": "synthetic_clean",
                "is_ground_truth": True,
                "split": "train",
            }
        )
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return rows_path


def test_trainable_no_reference_detector_scores_dirty_and_clean_minus(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    rows_path = _write_contamination_rows(tmp_path)
    model_dir = tmp_path / "model"

    manifest = train_stem_contamination_detector(
        rows_jsonl=rows_path,
        output_dir=model_dir,
        device="cpu",
        epochs=120,
        hidden_size=8,
        min_positive_rows=2,
    )

    assert manifest["training_status"] == "trained"
    assert manifest["detector_mode"] == "no_reference_neural_v1"
    assert manifest["shadow_only"] is True
    assert Path(manifest["model_file"]).exists()

    dirty_minus = sine(110.0, amp=0.18) + sine(880.0, amp=0.05)
    dirty_result = score_stem_contamination(
        tested_stem_path=write_wav(tmp_path / "dirty_minus.wav", dirty_minus),
        tested_stem="minus",
        foreign_source="vocals",
        output_dir=tmp_path / "dirty_score",
        foreign_reference_path=None,
        model_manifest_path=model_dir / "manifest.json",
    )

    clean_minus = sine(110.0, amp=0.18) + sine(220.0, amp=0.04)
    clean_result = score_stem_contamination(
        tested_stem_path=write_wav(tmp_path / "clean_minus.wav", clean_minus),
        tested_stem="minus",
        foreign_source="vocals",
        output_dir=tmp_path / "clean_score",
        foreign_reference_path=None,
        model_manifest_path=model_dir / "manifest.json",
    )

    assert dirty_result["verdict"] == "fail"
    assert dirty_result["detector_mode"] == "no_reference_neural_v1"
    assert dirty_result["leak_audio"]
    assert clean_result["verdict"] == "pass"
    assert clean_result["detector_mode"] == "no_reference_neural_v1"
    assert clean_result["leak_audio"] == ""


def test_trainable_no_reference_detector_scores_non_minus_pair(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    rows = []
    for idx, freq in enumerate((820.0, 880.0, 960.0), start=1):
        clean_guitar = sine(330.0 + idx * 20.0, amp=0.14) + sine(440.0, amp=0.03)
        leaked_vocal = sine(freq, amp=0.05)
        row_id = f"d{idx:06d}"
        rows.append(
            {
                "row_id": row_id,
                "input_stem": str(write_wav(tmp_path / "train" / f"{row_id}_in.flac", clean_guitar + leaked_vocal)),
                "target_leak": str(write_wav(tmp_path / "train" / f"{row_id}_leak.flac", leaked_vocal)),
                "base_stem": "guitar",
                "foreign_source": "vocals",
                "label_type": "synthetic_clean",
                "is_ground_truth": True,
                "split": "train",
            }
        )
    rows_path = tmp_path / "rows_guitar_vocals.jsonl"
    rows_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    model_dir = tmp_path / "model_guitar_vocals"

    manifest = train_stem_contamination_detector(
        rows_jsonl=rows_path,
        output_dir=model_dir,
        device="cpu",
        epochs=120,
        hidden_size=8,
        min_positive_rows=2,
    )

    dirty_guitar = sine(330.0, amp=0.14) + sine(880.0, amp=0.05)
    clean_guitar = sine(330.0, amp=0.14) + sine(440.0, amp=0.03)
    dirty_result = score_stem_contamination(
        tested_stem_path=write_wav(tmp_path / "dirty_guitar.wav", dirty_guitar),
        tested_stem="guitar",
        foreign_source="vocals",
        output_dir=tmp_path / "dirty_guitar_score",
        foreign_reference_path=None,
        model_manifest_path=model_dir / "manifest.json",
    )
    clean_result = score_stem_contamination(
        tested_stem_path=write_wav(tmp_path / "clean_guitar.wav", clean_guitar),
        tested_stem="guitar",
        foreign_source="vocals",
        output_dir=tmp_path / "clean_guitar_score",
        foreign_reference_path=None,
        model_manifest_path=model_dir / "manifest.json",
    )

    assert manifest["training_status"] == "trained"
    assert any(
        pair["tested_stem"] == "guitar" and pair["foreign_source"] == "vocals"
        for pair in manifest["trained_pairs"]
    )
    assert dirty_result["verdict"] == "fail"
    assert dirty_result["detector_mode"] == "no_reference_neural_v1"
    assert dirty_result["leak_audio"]
    assert clean_result["verdict"] == "pass"
    assert clean_result["detector_mode"] == "no_reference_neural_v1"


def test_trainable_no_reference_detector_blocks_untrained_pair(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    rows_path = _write_contamination_rows(tmp_path)
    model_dir = tmp_path / "model"
    train_stem_contamination_detector(
        rows_jsonl=rows_path,
        output_dir=model_dir,
        device="cpu",
        epochs=120,
        hidden_size=8,
        min_positive_rows=2,
    )

    result = score_stem_contamination(
        tested_stem_path=write_wav(tmp_path / "guitar.wav", sine(330.0, amp=0.2) + sine(880.0, amp=0.05)),
        tested_stem="guitar",
        foreign_source="vocals",
        output_dir=tmp_path / "score",
        foreign_reference_path=None,
        model_manifest_path=model_dir / "manifest.json",
    )

    assert result["verdict"] == "blocked_unknown"
    assert "model_pair_untrained" in result["reason"]


def test_trainable_no_reference_detector_single_manifest_scores_multiple_pairs(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    rows = []
    pair_specs = [
        ("minus", "vocals", 110.0, 0.18),
        ("guitar", "vocals", 330.0, 0.14),
    ]
    for pair_idx, (base_stem, foreign_source, base_freq, base_amp) in enumerate(pair_specs, start=1):
        for sample_idx, leak_freq in enumerate((820.0, 880.0, 960.0), start=1):
            clean = sine(base_freq + sample_idx * 20.0, amp=base_amp) + sine(base_freq * 2.0, amp=0.025)
            leak = sine(leak_freq, amp=0.05)
            row_id = f"d{pair_idx}{sample_idx:05d}"
            rows.append(
                {
                    "row_id": row_id,
                    "input_stem": str(write_wav(tmp_path / "train_multi" / f"{row_id}_in.flac", clean + leak)),
                    "target_leak": str(write_wav(tmp_path / "train_multi" / f"{row_id}_leak.flac", leak)),
                    "base_stem": base_stem,
                    "foreign_source": foreign_source,
                    "label_type": "synthetic_clean",
                    "is_ground_truth": True,
                    "split": "train",
                }
            )
    rows_path = tmp_path / "rows_multi.jsonl"
    rows_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    model_dir = tmp_path / "model_multi"

    manifest = train_stem_contamination_detector(
        rows_jsonl=rows_path,
        output_dir=model_dir,
        device="cpu",
        epochs=160,
        hidden_size=12,
        min_positive_rows=4,
    )

    dirty_minus = score_stem_contamination(
        tested_stem_path=write_wav(tmp_path / "dirty_minus.wav", sine(110.0, amp=0.18) + sine(880.0, amp=0.05)),
        tested_stem="minus",
        foreign_source="vocals",
        output_dir=tmp_path / "dirty_minus_multi_score",
        foreign_reference_path=None,
        model_manifest_path=model_dir / "manifest.json",
    )
    dirty_guitar = score_stem_contamination(
        tested_stem_path=write_wav(tmp_path / "dirty_guitar.wav", sine(330.0, amp=0.14) + sine(880.0, amp=0.05)),
        tested_stem="guitar",
        foreign_source="vocals",
        output_dir=tmp_path / "dirty_guitar_multi_score",
        foreign_reference_path=None,
        model_manifest_path=model_dir / "manifest.json",
    )

    trained_pairs = {(pair["tested_stem"], pair["foreign_source"]) for pair in manifest["trained_pairs"]}
    assert manifest["training_status"] == "trained"
    assert manifest["pair_count"] == 2
    assert trained_pairs == {("minus", "vocals"), ("guitar", "vocals")}
    assert dirty_minus["verdict"] == "fail"
    assert dirty_guitar["verdict"] == "fail"


def test_full_matrix_detector_treats_clean_voice_like_vocal_as_negative(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    clean_pairs_manifest = build_procedural_stem_clean_pairs(
        output_dir=tmp_path / "pairs",
        sample_rate=SAMPLE_RATE,
        duration_sec=0.75,
        examples_per_pair=6,
    )
    rows_path = build_dataset_from_manifest(clean_pairs_manifest, tmp_path / "dataset", contamination_db=-24.0)
    model_dir = tmp_path / "full_matrix_model"
    train_stem_contamination_detector(
        rows_jsonl=rows_path,
        output_dir=model_dir,
        device="cpu",
        epochs=220,
        hidden_size=24,
        min_positive_rows=6,
    )

    clean_vocal_path = write_wav(tmp_path / "clean_voice_like_vocal.flac", voice_like_vocal())
    for foreign_source in ("guitar", "piano", "other"):
        result = score_stem_contamination(
            tested_stem_path=clean_vocal_path,
            tested_stem="vocals",
            foreign_source=foreign_source,
            output_dir=tmp_path / "clean_voice_score",
            foreign_reference_path=None,
            model_manifest_path=model_dir / "manifest.json",
        )

        assert result["verdict"] == "pass", result
        assert result["leak_audio"] == ""


def test_full_matrix_detector_keeps_bright_clean_vocal_but_blocks_injected_guitar(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    clean_pairs_manifest = build_procedural_stem_clean_pairs(
        output_dir=tmp_path / "pairs",
        sample_rate=SAMPLE_RATE,
        duration_sec=0.75,
        examples_per_pair=12,
    )
    rows_path = build_dataset_from_manifest(clean_pairs_manifest, tmp_path / "dataset", contamination_db=-24.0)
    model_dir = tmp_path / "bright_vocal_model"
    train_stem_contamination_detector(
        rows_jsonl=rows_path,
        output_dir=model_dir,
        device="cpu",
        epochs=260,
        hidden_size=32,
        min_positive_rows=12,
    )

    clean_vocal_path = write_wav(tmp_path / "bright_clean_vocal.flac", bright_voice_like_vocal())
    for foreign_source in ("drums", "bass", "guitar", "piano", "other"):
        result = score_stem_contamination(
            tested_stem_path=clean_vocal_path,
            tested_stem="vocals",
            foreign_source=foreign_source,
            output_dir=tmp_path / "bright_clean_score",
            foreign_reference_path=None,
            model_manifest_path=model_dir / "manifest.json",
        )
        assert result["verdict"] == "pass", result
        assert result["leak_audio"] == ""

    dirty_result = score_stem_contamination(
        tested_stem_path=write_wav(
            tmp_path / "bright_vocal_with_guitar.flac",
            bright_voice_like_vocal() + sustained_mid_tone(440.0, amp=0.16),
        ),
        tested_stem="vocals",
        foreign_source="guitar",
        output_dir=tmp_path / "bright_dirty_score",
        foreign_reference_path=None,
        model_manifest_path=model_dir / "manifest.json",
    )
    assert dirty_result["verdict"] == "fail", dirty_result
    assert dirty_result["leak_audio"]


def test_full_matrix_detector_treats_clean_kick_drum_as_negative_for_bass(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    clean_pairs_manifest = build_procedural_stem_clean_pairs(
        output_dir=tmp_path / "pairs",
        sample_rate=SAMPLE_RATE,
        duration_sec=0.75,
        examples_per_pair=8,
    )
    rows_path = build_dataset_from_manifest(clean_pairs_manifest, tmp_path / "dataset", contamination_db=-24.0)
    model_dir = tmp_path / "kick_model"
    train_stem_contamination_detector(
        rows_jsonl=rows_path,
        output_dir=model_dir,
        device="cpu",
        epochs=220,
        hidden_size=24,
        min_positive_rows=8,
    )

    result = score_stem_contamination(
        tested_stem_path=write_wav(tmp_path / "clean_kick.flac", kick_drum_like()),
        tested_stem="drums",
        foreign_source="bass",
        output_dir=tmp_path / "clean_kick_score",
        foreign_reference_path=None,
        model_manifest_path=model_dir / "manifest.json",
    )

    assert result["verdict"] == "pass", result
    assert result["leak_audio"] == ""


def test_full_matrix_detector_separates_boomy_kick_from_sustained_bass_leak(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    clean_pairs_manifest = build_procedural_stem_clean_pairs(
        output_dir=tmp_path / "pairs",
        sample_rate=SAMPLE_RATE,
        duration_sec=0.75,
        examples_per_pair=10,
    )
    rows_path = build_dataset_from_manifest(clean_pairs_manifest, tmp_path / "dataset", contamination_db=-24.0)
    model_dir = tmp_path / "boom_model"
    train_stem_contamination_detector(
        rows_jsonl=rows_path,
        output_dir=model_dir,
        device="cpu",
        epochs=220,
        hidden_size=24,
        min_positive_rows=10,
    )

    clean_result = score_stem_contamination(
        tested_stem_path=write_wav(tmp_path / "boomy_clean_kick.flac", boomy_kick_drum_like()),
        tested_stem="drums",
        foreign_source="bass",
        output_dir=tmp_path / "clean_boom_score",
        foreign_reference_path=None,
        model_manifest_path=model_dir / "manifest.json",
    )
    dirty_result = score_stem_contamination(
        tested_stem_path=write_wav(
            tmp_path / "boomy_kick_with_sustained_bass.flac",
            boomy_kick_drum_like() + sustained_bass_tone(),
        ),
        tested_stem="drums",
        foreign_source="bass",
        output_dir=tmp_path / "dirty_boom_score",
        foreign_reference_path=None,
        model_manifest_path=model_dir / "manifest.json",
    )

    assert clean_result["verdict"] == "pass", clean_result
    assert clean_result["leak_audio"] == ""
    assert dirty_result["verdict"] == "fail", dirty_result
    assert dirty_result["leak_audio"]


def test_full_matrix_detector_keeps_bass_harmonics_but_blocks_injected_guitar(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    clean_pairs_manifest = build_procedural_stem_clean_pairs(
        output_dir=tmp_path / "pairs",
        sample_rate=SAMPLE_RATE,
        duration_sec=0.75,
        examples_per_pair=12,
    )
    rows_path = build_dataset_from_manifest(clean_pairs_manifest, tmp_path / "dataset", contamination_db=-24.0)
    model_dir = tmp_path / "bass_harmonic_model"
    train_stem_contamination_detector(
        rows_jsonl=rows_path,
        output_dir=model_dir,
        device="cpu",
        epochs=260,
        hidden_size=32,
        min_positive_rows=12,
    )

    clean_bass_path = write_wav(tmp_path / "harmonic_clean_bass.flac", harmonic_bass_like())
    for foreign_source in ("guitar", "piano", "other"):
        result = score_stem_contamination(
            tested_stem_path=clean_bass_path,
            tested_stem="bass",
            foreign_source=foreign_source,
            output_dir=tmp_path / "clean_bass_score",
            foreign_reference_path=None,
            model_manifest_path=model_dir / "manifest.json",
        )
        assert result["verdict"] == "pass", result
        assert result["leak_audio"] == ""

    dirty_result = score_stem_contamination(
        tested_stem_path=write_wav(
            tmp_path / "bass_with_guitar.flac",
            harmonic_bass_like() + sustained_mid_tone(440.0, amp=0.14),
        ),
        tested_stem="bass",
        foreign_source="guitar",
        output_dir=tmp_path / "dirty_bass_score",
        foreign_reference_path=None,
        model_manifest_path=model_dir / "manifest.json",
    )
    assert dirty_result["verdict"] == "fail", dirty_result
    assert dirty_result["leak_audio"]


def test_full_matrix_detector_keeps_clean_guitar_but_blocks_injected_piano(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    clean_pairs_manifest = build_procedural_stem_clean_pairs(
        output_dir=tmp_path / "pairs",
        sample_rate=SAMPLE_RATE,
        duration_sec=0.75,
        examples_per_pair=12,
    )
    rows_path = build_dataset_from_manifest(clean_pairs_manifest, tmp_path / "dataset", contamination_db=-24.0)
    model_dir = tmp_path / "guitar_model"
    train_stem_contamination_detector(
        rows_jsonl=rows_path,
        output_dir=model_dir,
        device="cpu",
        epochs=260,
        hidden_size=32,
        min_positive_rows=12,
    )

    clean_guitar_path = write_wav(tmp_path / "clean_guitar.flac", guitar_like_stem())
    for foreign_source in ("vocals", "drums", "bass", "piano", "other"):
        result = score_stem_contamination(
            tested_stem_path=clean_guitar_path,
            tested_stem="guitar",
            foreign_source=foreign_source,
            output_dir=tmp_path / "clean_guitar_score",
            foreign_reference_path=None,
            model_manifest_path=model_dir / "manifest.json",
        )
        assert result["verdict"] == "pass", result
        assert result["leak_audio"] == ""

    dirty_result = score_stem_contamination(
        tested_stem_path=write_wav(
            tmp_path / "guitar_with_piano.flac",
            guitar_like_stem() + sustained_mid_tone(880.0, amp=0.14),
        ),
        tested_stem="guitar",
        foreign_source="piano",
        output_dir=tmp_path / "dirty_guitar_score",
        foreign_reference_path=None,
        model_manifest_path=model_dir / "manifest.json",
    )
    assert dirty_result["verdict"] == "fail", dirty_result
    assert dirty_result["leak_audio"]


def test_full_matrix_detector_separates_cymbal_drums_from_sustained_melodic_leak(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    clean_pairs_manifest = build_procedural_stem_clean_pairs(
        output_dir=tmp_path / "pairs",
        sample_rate=SAMPLE_RATE,
        duration_sec=0.75,
        examples_per_pair=10,
    )
    rows_path = build_dataset_from_manifest(clean_pairs_manifest, tmp_path / "dataset", contamination_db=-24.0)
    model_dir = tmp_path / "cymbal_model"
    train_stem_contamination_detector(
        rows_jsonl=rows_path,
        output_dir=model_dir,
        device="cpu",
        epochs=220,
        hidden_size=24,
        min_positive_rows=10,
    )

    clean_path = write_wav(tmp_path / "cymbal_clean_drums.flac", cymbal_like_drums())
    for foreign_source in ("guitar", "piano", "other"):
        clean_result = score_stem_contamination(
            tested_stem_path=clean_path,
            tested_stem="drums",
            foreign_source=foreign_source,
            output_dir=tmp_path / "clean_cymbal_score",
            foreign_reference_path=None,
            model_manifest_path=model_dir / "manifest.json",
        )
        assert clean_result["verdict"] == "pass", clean_result
        assert clean_result["leak_audio"] == ""

    dirty_result = score_stem_contamination(
        tested_stem_path=write_wav(
            tmp_path / "cymbal_with_sustained_guitar.flac",
            cymbal_like_drums() + sustained_mid_tone(440.0),
        ),
        tested_stem="drums",
        foreign_source="guitar",
        output_dir=tmp_path / "dirty_cymbal_score",
        foreign_reference_path=None,
        model_manifest_path=model_dir / "manifest.json",
    )

    assert dirty_result["verdict"] == "fail", dirty_result
    assert dirty_result["leak_audio"]
