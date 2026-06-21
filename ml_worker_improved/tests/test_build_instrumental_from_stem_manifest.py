from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from build_instrumental_from_stem_manifest import (  # noqa: E402
    build_instrumental,
    parse_stem_weights,
    resolve_manifest_stems,
)


SAMPLE_RATE = 16_000


def write_stem(path: Path, value: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.full((SAMPLE_RATE // 10, 2), value, dtype=np.float32)
    sf.write(path, audio, SAMPLE_RATE)
    return path


def test_build_instrumental_preserves_default_unweighted_sum(tmp_path: Path) -> None:
    drums = write_stem(tmp_path / "drums.wav", 0.10)
    guitar = write_stem(tmp_path / "guitar.wav", 0.20)

    instrumental, sample_rate, summary = build_instrumental([("drums", drums), ("guitar", guitar)])

    assert sample_rate == SAMPLE_RATE
    assert np.allclose(instrumental, 0.30, atol=1.0e-4)
    assert summary["stem_weights"] == {"drums": 1.0, "guitar": 1.0}
    assert summary["included_stems"] == ["drums", "guitar"]


def test_build_instrumental_applies_named_stem_weights(tmp_path: Path) -> None:
    drums = write_stem(tmp_path / "drums.wav", 0.10)
    guitar = write_stem(tmp_path / "guitar.wav", 0.20)

    instrumental, _sample_rate, summary = build_instrumental(
        [("drums", drums), ("guitar", guitar)],
        stem_weights={"guitar": 0.25},
    )

    assert np.allclose(instrumental, 0.15, atol=1.0e-4)
    assert summary["stem_weights"] == {"drums": 1.0, "guitar": 0.25}


def test_build_instrumental_can_clip_peaks_instead_of_normalizing(tmp_path: Path) -> None:
    drums = write_stem(tmp_path / "drums.wav", 0.80)
    guitar = write_stem(tmp_path / "guitar.wav", 0.50)

    instrumental, _sample_rate, summary = build_instrumental(
        [("drums", drums), ("guitar", guitar)],
        peak_mode="clip",
    )

    assert np.max(instrumental) <= 1.0
    assert np.allclose(instrumental, 1.0, atol=1.0e-4)
    assert summary["peak_mode"] == "clip"
    assert summary["normalization_gain"] == 1.0


def test_parse_stem_weights_rejects_invalid_weight() -> None:
    with pytest.raises(ValueError, match="Invalid stem weight"):
        parse_stem_weights(["guitar=-0.25"])


def test_resolve_manifest_stems_rejects_weight_for_missing_stem(tmp_path: Path) -> None:
    manifest = {
        "stems": {
            "drums": {"path": str(write_stem(tmp_path / "drums.wav", 0.10))},
        }
    }

    with pytest.raises(ValueError, match="Stem weight references missing stem"):
        resolve_manifest_stems(manifest, excluded_stems=set(), stem_weights={"guitar": 0.5})
