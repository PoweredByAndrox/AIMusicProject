from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "apply_actual_masked_residue_minus_cleanup.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("apply_actual_masked_residue_minus_cleanup", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_low_priority_windows_require_explicit_flag_and_score() -> None:
    module = _load_module()
    windows = [
        {
            "start_sec": 0.0,
            "end_sec": 1.0,
            "vocal_evidence_status": "low_priority",
            "suspicion_score": 0.15,
        },
        {
            "start_sec": 1.0,
            "end_sec": 2.0,
            "vocal_evidence_status": "low_priority",
            "suspicion_score": 0.05,
        },
    ]

    default_gate = module.build_review_window_gate(
        windows=windows,
        frame_count=200,
        sample_rate=100,
        smoothing_sec=0.0,
    )
    low_priority_gate = module.build_review_window_gate(
        windows=windows,
        frame_count=200,
        sample_rate=100,
        smoothing_sec=0.0,
        include_low_priority_windows=True,
        min_low_priority_score=0.12,
    )

    assert float(default_gate.max()) == 0.0
    assert float(low_priority_gate[:100].max()) == 1.0
    assert float(low_priority_gate[100:].max()) == 0.0


def test_base_overlap_power_reduces_broad_residue_mask() -> None:
    module = _load_module()
    sample_rate = 8000
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    vocal = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    base = (vocal + 0.8 * np.sin(2.0 * np.pi * 880.0 * t)).astype(np.float32)
    review_gate = np.ones(sample_rate, dtype=np.float32)

    _, ungated = module.build_actual_residue_gain_mask(
        base_minus=base,
        primary_vocals=vocal,
        review_gate=review_gate,
        sample_rate=sample_rate,
        strength=0.1,
        mask_power=1.0,
        base_overlap_power=0.0,
    )
    _, overlap_gated = module.build_actual_residue_gain_mask(
        base_minus=base,
        primary_vocals=vocal,
        review_gate=review_gate,
        sample_rate=sample_rate,
        strength=0.1,
        mask_power=1.0,
        base_overlap_power=1.0,
    )

    assert overlap_gated["mean_residue_mask"] < ungated["mean_residue_mask"]


def _write_audio(path: Path, sample_rate: int = 12000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = sample_rate // 2
    t = np.linspace(0.0, 0.5, frames, endpoint=False, dtype=np.float32)
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

    module.main(
        [
            str(input_dir / "raw.flac"),
            str(input_dir / "base.flac"),
            str(input_dir / "guide.flac"),
            str(input_dir / "primary.flac"),
            str(input_dir / "secondary.flac"),
            str(output_dir),
            "--listen-dir",
            str(listen_dir),
        ]
    )
    manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))

    assert (output_dir / "raw_mix.flac").is_file()
    assert (output_dir / "base_product_minus.flac").is_file()
    assert (output_dir / "residue_product_minus.flac").is_file()
    assert not (output_dir / "raw_ref.flac").exists()
    assert not (output_dir / "minus_base.flac").exists()
    assert not (output_dir / "minus_actual_masked_residue.flac").exists()
    assert manifest["stem_type"] == "minus"
    assert manifest["artifact_role"] == "candidate_product_minus"
    assert manifest["candidate_product_artifact_key"] == "residue_product_minus_flac"
    assert manifest["artifacts"]["residue_product_minus_flac"].endswith("residue_product_minus.flac")
    assert manifest["artifacts"]["minus_actual_masked_residue_flac"].endswith("residue_product_minus.flac")
    assert (listen_dir / "00_raw_mix.flac").is_file()
    assert (listen_dir / "01_base_product_minus.flac").is_file()
    assert (listen_dir / "02_candidate_product_minus.flac").is_file()
