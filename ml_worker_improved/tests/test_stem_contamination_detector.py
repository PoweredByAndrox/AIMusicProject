from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.build_stem_contamination_dataset import validate_stem_contamination_row
from tools.score_stem_contamination import score_stem_contamination
from tools.stem_contamination_common import build_minus_from_non_vocal_stems
from tools.train_stem_contamination_detector import train_stem_contamination_detector


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float = 0.25, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


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
