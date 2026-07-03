from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "apply_windowed_hybrid_minus_cleanup.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("apply_windowed_hybrid_minus_cleanup", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frame_score_gate_recovers_sparse_max_percentile_gate() -> None:
    module = _load_module()
    frame_score = np.zeros(64, dtype=np.float32)
    frame_score[[10, 24, 48]] = 1.0

    gate, metrics = module._frame_score_gate(
        frame_score,
        score_percentile=100.0,
        smooth_width=9,
        apply_threshold=0.05,
    )

    assert metrics["gate_mode"] in {"continuous_max_guard", "continuous_sparse_recovery"}
    assert metrics["frame_score_max"] == 1.0
    assert float(gate.max()) > 0.0
    assert float(gate.mean()) > 0.0


def test_frame_score_gate_stays_zero_without_positive_scores() -> None:
    module = _load_module()
    frame_score = np.zeros(32, dtype=np.float32)

    gate, metrics = module._frame_score_gate(
        frame_score,
        score_percentile=70.0,
        smooth_width=9,
        apply_threshold=0.14,
    )

    assert metrics["gate_mode"] == "empty_score"
    assert float(gate.max()) == 0.0


def test_guide_floor_is_mask_and_activity_gated() -> None:
    module = _load_module()
    guide = np.zeros((2, 4), dtype=np.float32)
    cleanup_mask = np.array(
        [
            [1.0, 0.5, 0.0, 1.0],
            [0.25, 0.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )
    time_activity = np.array([1.0, 0.5, 1.0, 0.0], dtype=np.float32)

    floored = module._apply_guide_floor(
        guide,
        cleanup_mask,
        time_activity,
        guide_floor=0.2,
    )

    assert np.isclose(floored[0, 0], 0.2)
    assert np.isclose(floored[0, 1], 0.05)
    assert np.isclose(floored[0, 2], 0.0)
    assert np.isclose(floored[1, 2], 0.2)
    assert np.isclose(floored[:, 3].max(), 0.0)


def _write_audio(path: Path, sample_rate: int = 12000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = sample_rate * 4
    t = np.linspace(0.0, 4.0, frames, endpoint=False, dtype=np.float32)
    audio = np.column_stack(
        [
            0.05 * np.sin(2.0 * np.pi * 440.0 * t),
            0.05 * np.sin(2.0 * np.pi * 660.0 * t),
        ]
    ).astype(np.float32)
    sf.write(path, audio, sample_rate, subtype="PCM_24")


def test_main_writes_role_explicit_candidate_artifact_names(tmp_path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "require_docker_runtime", lambda **_kwargs: None)
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    listen_dir = tmp_path / "listen"
    for name in ("raw", "base", "guide", "primary", "secondary"):
        _write_audio(input_dir / f"{name}.flac")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_windowed_hybrid_minus_cleanup.py",
            str(input_dir / "raw.flac"),
            str(input_dir / "base.flac"),
            str(input_dir / "guide.flac"),
            str(input_dir / "primary.flac"),
            str(input_dir / "secondary.flac"),
            str(output_dir),
            "--listen-dir",
            str(listen_dir),
        ],
    )

    module.main()
    manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))

    assert (output_dir / "raw_mix.flac").is_file()
    assert (output_dir / "base_product_minus.flac").is_file()
    assert (output_dir / "guide_product_minus.flac").is_file()
    assert (output_dir / "windowed_product_minus.flac").is_file()
    assert not (output_dir / "raw_ref.flac").exists()
    assert not (output_dir / "minus_base.flac").exists()
    assert not (output_dir / "minus_guide.flac").exists()
    assert not (output_dir / "minus_windowed_hybrid.flac").exists()
    assert manifest["stem_type"] == "minus"
    assert manifest["artifact_role"] == "candidate_product_minus"
    assert manifest["candidate_product_artifact_key"] == "windowed_product_minus_flac"
    assert manifest["artifacts"]["windowed_product_minus_flac"].endswith("windowed_product_minus.flac")
    assert manifest["artifacts"]["minus_windowed_hybrid_flac"].endswith("windowed_product_minus.flac")
    assert (listen_dir / "00_raw_mix.flac").is_file()
    assert (listen_dir / "01_base_product_minus.flac").is_file()
    assert (listen_dir / "02_candidate_product_minus.flac").is_file()
