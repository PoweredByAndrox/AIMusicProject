from __future__ import annotations

import numpy as np

from tools.mac_audio_damage_audit import score_instrumental_damage_from_arrays
from tools.mac_audio_damage_audit import run_damage_audit


def _tone(sample_rate: int, seconds: float, freqs: list[float]) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
    mono = sum(0.08 * np.sin(2.0 * np.pi * freq * t) for freq in freqs).astype(np.float32)
    return np.stack([mono, mono], axis=1)


def test_region_detail_loss_blocks_even_when_global_loudness_is_similar() -> None:
    sample_rate = 16000
    reference = _tone(sample_rate, 2.0, [220.0, 440.0, 6500.0])
    candidate = reference.copy()
    start = int(0.75 * sample_rate)
    end = int(1.25 * sample_rate)
    candidate[start:end] *= 1.25
    candidate[start:end] += _tone(sample_rate, 0.5, [220.0, 440.0]) * 0.2
    candidate[start:end] -= _tone(sample_rate, 0.5, [6500.0]) * 0.95

    result = score_instrumental_damage_from_arrays(
        reference_audio=reference,
        candidate_audio=candidate,
        sample_rate=sample_rate,
        regions=[{"start": 0.75, "end": 1.25}],
    )

    assert result["verdict"] == "fail"
    assert "instrumental_region_detail_loss" in result["blocker_reasons"]


def test_clean_candidate_passes_damage_audit() -> None:
    sample_rate = 16000
    reference = _tone(sample_rate, 2.0, [220.0, 440.0, 6500.0])
    candidate = reference * 0.995

    result = score_instrumental_damage_from_arrays(
        reference_audio=reference,
        candidate_audio=candidate,
        sample_rate=sample_rate,
        regions=[{"start": 0.5, "end": 1.0}],
    )

    assert result["verdict"] == "pass"
    assert result["blocker_reasons"] == []


def test_file_damage_audit_blocks_sample_rate_mismatch(tmp_path) -> None:
    import json
    import soundfile as sf

    reference = _tone(16000, 1.0, [220.0, 6500.0])
    candidate = _tone(8000, 1.0, [220.0, 6500.0])
    reference_path = tmp_path / "ref.flac"
    candidate_path = tmp_path / "cand.flac"
    score_path = tmp_path / "pitch.json"
    sf.write(reference_path, reference, 16000)
    sf.write(candidate_path, candidate, 8000)
    score_path.write_text(json.dumps({"regions": [{"start": 0.1, "end": 0.4}]}), encoding="utf-8")

    result = run_damage_audit(
        reference_instrumental_path=reference_path,
        candidate_instrumental_path=candidate_path,
        score_json_path=score_path,
        output_dir=tmp_path / "out",
    )

    assert result["verdict"] == "blocked_unknown"
    assert result["blocker_reasons"] == ["damage_audit_sample_rate_mismatch"]
    assert result["delivery_allowed"] is False
    assert (tmp_path / "out" / "damage.json").exists()
