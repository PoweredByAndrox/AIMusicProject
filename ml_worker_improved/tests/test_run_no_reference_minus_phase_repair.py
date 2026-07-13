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


def _rms_delta(source: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(candidate - source))))


def _correlation(source: np.ndarray, candidate: np.ndarray) -> float:
    source_flat = source.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    return float(np.corrcoef(source_flat, candidate_flat)[0, 1])


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
    expected_variants = [
        ("minus_mid_mono", 0.0),
        ("minus_side_attenuate_25", 0.25),
        ("minus_side_attenuate_45", 0.45),
        ("minus_side_attenuate_46", 0.46),
        ("minus_side_attenuate_47", 0.47),
        ("minus_side_attenuate_50", 0.50),
    ]
    assert manifest["variant_count"] == len(expected_variants)
    assert len(rows) == len(expected_variants)
    assert manifest["track_id"] == "track-a"
    assert manifest["source_candidate_id"] == "candidate-a"
    assert manifest["source_minus_route_id"] == "source-minus"
    assert manifest["source_mix_path"] == str(mix_path)
    assert manifest["source_vocal_path"] == str(vocal_path)
    assert manifest["source_minus_path"] == str(minus_path)
    assert manifest["delivery_allowed"] is False
    assert manifest["delivery_status"] == "diagnostic_only"
    assert manifest["presentation_status"] == "phase_repair_candidate_not_listen_pack"
    assert manifest["release_status"] == "not_release_ready"
    assert manifest["release_ready"] is False
    assert manifest["gpu_inference_skipped"] is True
    assert [(item["variant"], item["side_gain"]) for item in manifest["variants"]] == expected_variants
    assert [row["minus_phase_repair_variant"] for row in rows] == [name for name, _ in expected_variants]
    assert all(row["route_contract"] == "no_reference_minus_phase_repair_candidate" for row in rows)
    assert all(row["delivery_status"] == "diagnostic_only" for row in rows)
    assert all(row["delivery_allowed"] is False for row in rows)
    assert all(row["presentation_status"] == "phase_repair_candidate_not_listen_pack" for row in rows)
    assert all(row["release_status"] == "not_release_ready" for row in rows)
    assert all(row["release_ready"] is False for row in rows)
    assert all(row["can_publish_as_release"] is False for row in rows)
    assert all(row["can_mark_release_ready"] is False for row in rows)
    assert all(row["track_id"] == "track-a" for row in rows)
    assert all(row["mix_path"] == str(mix_path) for row in rows)
    assert all(row["vocal_path"] == str(vocal_path) for row in rows)
    assert all(row["candidate_id"] == f"candidate-a__{row['minus_phase_repair_variant']}" for row in rows)
    assert all(row["instrumental_path"] == row["minus_path"] for row in rows)
    assert all(row["minus_route_id"] == f"source-minus__{row['minus_phase_repair_variant']}" for row in rows)

    rendered = []
    for manifest_variant, row, (variant_name, side_gain) in zip(manifest["variants"], rows, expected_variants):
        assert manifest_variant["variant"] == variant_name
        assert manifest_variant["side_gain"] == side_gain
        assert manifest_variant["path"] == row["minus_path"]
        assert Path(row["minus_path"]).exists()
        audio, rendered_rate = sf.read(row["minus_path"], always_2d=True, dtype="float32")
        assert rendered_rate == sample_rate
        rendered.append((side_gain, audio))

    phase_values = [module.phase_consistency(audio) for _, audio in rendered]
    assert np.all(np.diff(phase_values) < -1e-6)
    assert phase_values[0] > module.phase_consistency(minus)

    integrity_rms_deltas = [_rms_delta(minus, audio) for _, audio in rendered]
    integrity_correlations = [_correlation(minus, audio) for _, audio in rendered]
    assert np.all(np.diff(integrity_rms_deltas) <= 1e-5)
    assert np.all(np.diff(integrity_correlations) >= -1e-5)


def test_build_minus_phase_repair_aligns_mismatched_sample_rates(tmp_path: Path) -> None:
    module = _load_module()
    mix_rate = 48000
    stem_rate = 44100
    mix_t = np.linspace(0.0, 1.0, mix_rate, endpoint=False)
    stem_t = np.linspace(0.0, 1.0, stem_rate, endpoint=False)
    mix = np.column_stack([0.2 * np.sin(2.0 * np.pi * 220.0 * mix_t)] * 2)
    vocal = np.column_stack([0.05 * np.sin(2.0 * np.pi * 440.0 * stem_t)] * 2)
    minus = np.column_stack([0.25 * np.sin(2.0 * np.pi * 330.0 * stem_t)] * 2)
    mix_path = tmp_path / "mix48.wav"
    vocal_path = tmp_path / "vocal441.wav"
    minus_path = tmp_path / "minus441.wav"
    _write_stereo(mix_path, mix, mix_rate)
    _write_stereo(vocal_path, vocal, stem_rate)
    _write_stereo(minus_path, minus, stem_rate)

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
    repaired, repaired_rate = sf.read(rows[0]["minus_path"], always_2d=True, dtype="float32")
    assert manifest["variant_count"] == 6
    assert len(rows) == 6
    assert repaired_rate == stem_rate
    assert repaired.shape[0] == stem_rate
    assert manifest["sample_rate_alignment"] == {
        "target_sample_rate": stem_rate,
        "mix_original_sample_rate": mix_rate,
        "vocal_original_sample_rate": stem_rate,
        "minus_original_sample_rate": stem_rate,
        "mix_resampled_to_target": True,
        "vocal_resampled_to_target": False,
        "minus_resampled_to_target": False,
    }
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False


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
