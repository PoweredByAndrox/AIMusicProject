from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "run_no_reference_minus_phase_repair.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_no_reference_minus_phase_repair", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_stereo(path: Path, audio: np.ndarray, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate, subtype="FLOAT")


def test_build_minus_phase_repair_writes_variants_and_candidate_pool(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 44100
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False)
    mid = 0.35 * np.sin(2.0 * np.pi * 220.0 * t)
    side = 0.30 * np.sin(2.0 * np.pi * 733.0 * t)
    minus = np.column_stack([mid + side, mid - side])
    mix = np.column_stack([mid + side * 0.7, mid - side * 0.7])
    vocal = np.column_stack([0.05 * np.sin(2.0 * np.pi * 440.0 * t)] * 2)
    mix_path = tmp_path / "mix.wav"
    vocal_path = tmp_path / "vocal.wav"
    minus_path = tmp_path / "minus.wav"
    _write_stereo(mix_path, mix, sample_rate)
    _write_stereo(vocal_path, vocal, sample_rate)
    _write_stereo(minus_path, minus, sample_rate)

    manifest = module.build_minus_phase_repair(
        mix_path=mix_path,
        vocal_path=vocal_path,
        minus_path=minus_path,
        output_dir=tmp_path / "repair",
        track_id="track-a",
        candidate_id="candidate-a",
        minus_route_id="source-minus",
    )

    rows = [
        json.loads(line)
        for line in Path(manifest["candidate_pair_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest["variant_count"] == 3
    assert manifest["delivery_allowed"] is False
    assert manifest["release_status"] == "not_release_ready"
    assert manifest["gpu_inference_skipped"] is True
    assert {row["minus_phase_repair_variant"] for row in rows} == {
        "minus_mid_mono",
        "minus_side_attenuate_25",
        "minus_side_attenuate_50",
    }
    assert all(row["route_contract"] == "no_reference_minus_phase_repair_candidate" for row in rows)
    assert all(row["delivery_allowed"] is False for row in rows)
    assert all(row["release_ready"] is False for row in rows)
    attenuated = sf.read(rows[1]["minus_path"], always_2d=True, dtype="float32")[0]
    assert module.phase_consistency(attenuated) > module.phase_consistency(minus)


def test_main_uses_docker_guard_and_writes_manifest(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 16000
    t = np.linspace(0.0, 0.1, int(sample_rate * 0.1), endpoint=False)
    audio = np.column_stack([0.1 * np.sin(2.0 * np.pi * 200.0 * t)] * 2)
    mix_path = tmp_path / "mix.wav"
    vocal_path = tmp_path / "vocal.wav"
    minus_path = tmp_path / "minus.wav"
    _write_stereo(mix_path, audio, sample_rate)
    _write_stereo(vocal_path, audio * 0.1, sample_rate)
    _write_stereo(minus_path, audio, sample_rate)

    with patch.object(module, "require_docker_runtime") as guard:
        exit_code = module.main(
            [
                "--mix-path",
                str(mix_path),
                "--vocal-path",
                str(vocal_path),
                "--minus-path",
                str(minus_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--track-id",
                "track-a",
                "--candidate-id",
                "candidate-a",
                "--minus-route-id",
                "source-minus",
            ]
        )

    assert exit_code == 0
    guard.assert_called_once()
    assert (tmp_path / "out" / "minus_phase_repair_manifest.json").exists()
