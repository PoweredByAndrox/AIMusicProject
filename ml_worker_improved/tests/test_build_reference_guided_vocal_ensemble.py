from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = TOOLS_ROOT / "build_reference_guided_vocal_ensemble.py"


def _load_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    spec = importlib.util.spec_from_file_location("build_reference_guided_vocal_ensemble", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_nonnegative_ridge_weights_reconstruct_reference_mix() -> None:
    module = _load_module()
    candidate_a = np.array([[1.0, 0.0], [0.5, 0.2], [0.0, 1.0]], dtype=np.float32)
    candidate_b = np.array([[0.0, 1.0], [0.2, 0.5], [1.0, 0.0]], dtype=np.float32)
    reference = (0.75 * candidate_a) + (0.25 * candidate_b)

    weights = module.solve_nonnegative_ridge_weights(
        candidate_audio=[candidate_a, candidate_b],
        reference_audio=reference,
        ridge=1.0e-8,
    )

    assert np.allclose(weights, np.array([0.75, 0.25]), atol=1.0e-3)


def test_segmented_ensemble_improves_over_best_single_candidate() -> None:
    module = _load_module()
    first_half = np.ones((8, 2), dtype=np.float32)
    second_half = np.full((8, 2), 0.2, dtype=np.float32)
    reference = np.concatenate([first_half, second_half], axis=0)
    candidate_a = np.concatenate([first_half, np.zeros((8, 2), dtype=np.float32)], axis=0)
    candidate_b = np.concatenate([np.zeros((8, 2), dtype=np.float32), second_half], axis=0)

    result = module.build_segmented_ensemble_audio(
        candidate_audio=[candidate_a, candidate_b],
        reference_audio=reference,
        sample_rate=8,
        window_sec=1.0,
        overlap_sec=0.0,
        ridge=1.0e-8,
    )

    best_single_error = min(
        float(np.mean((candidate_a - reference) ** 2)),
        float(np.mean((candidate_b - reference) ** 2)),
    )
    ensemble_error = float(np.mean((result.audio - reference) ** 2))
    assert ensemble_error < best_single_error * 0.01
    assert result.window_count == 2


def test_spectral_band_ensemble_combines_frequency_specific_candidates() -> None:
    module = _load_module()
    sample_rate = 64
    frames = 64
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    low = (0.4 * np.sin(2.0 * np.pi * 4.0 * t)).astype(np.float32)[:, None]
    high = (0.4 * np.sin(2.0 * np.pi * 18.0 * t)).astype(np.float32)[:, None]
    candidate_low = np.repeat(low, 2, axis=1)
    candidate_high = np.repeat(high, 2, axis=1)
    reference = candidate_low + candidate_high

    result = module.build_segmented_spectral_band_ensemble_audio(
        candidate_audio=[candidate_low, candidate_high],
        reference_audio=reference,
        sample_rate=sample_rate,
        window_sec=1.0,
        overlap_sec=0.0,
        ridge=1.0e-8,
        band_edges_hz=[0.0, 10.0, 32.0],
        candidate_labels=["low", "high"],
    )

    assert float(np.mean((result.audio - reference) ** 2)) < 1.0e-6
    assert result.window_summaries[0]["band_weights"]["0.0-10.0"]["low"] > 0.9
    assert result.window_summaries[0]["band_weights"]["10.0-32.0"]["high"] > 0.9


def test_complex_spectral_band_ensemble_can_correct_phase_per_band() -> None:
    module = _load_module()
    sample_rate = 64
    frames = 64
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    reference = (0.4 * np.sin(2.0 * np.pi * 8.0 * t)).astype(np.float32)[:, None]
    candidate_phase_shifted = (0.4 * np.cos(2.0 * np.pi * 8.0 * t)).astype(np.float32)[:, None]
    reference = np.repeat(reference, 2, axis=1)
    candidate_phase_shifted = np.repeat(candidate_phase_shifted, 2, axis=1)

    result = module.build_segmented_spectral_band_complex_ensemble_audio(
        candidate_audio=[candidate_phase_shifted],
        reference_audio=reference,
        sample_rate=sample_rate,
        window_sec=1.0,
        overlap_sec=0.0,
        ridge=1.0e-8,
        band_edges_hz=[0.0, 32.0],
        candidate_labels=["phase_shifted"],
    )

    assert float(np.mean((result.audio - reference) ** 2)) < 1.0e-6
    assert result.window_summaries[0]["complex_band_weights"]["0.0-32.0"]["phase_shifted"]["magnitude"] > 0.9


def test_reference_guided_manifest_blocks_no_reference_delivery(tmp_path: Path) -> None:
    module = _load_module()
    vocals_path = tmp_path / "vocals.flac"
    instrumental_path = tmp_path / "instrumental.flac"
    vocals_path.write_text("stub vocals", encoding="utf-8")
    instrumental_path.write_text("stub instrumental", encoding="utf-8")

    manifest = module.build_reference_guided_manifest(
        source_audio_path=Path("/tracks/source.flac"),
        ground_truth_manifest=Path("/refs/ground_truth_manifest.json"),
        candidate_specs=[
            module.CandidateSpec(label="mdx23c", manifest_path=Path("/runs/mdx/stem_manifest.json")),
            module.CandidateSpec(label="melband", manifest_path=Path("/runs/melband/stem_manifest.json")),
        ],
        vocals_path=vocals_path,
        instrumental_path=instrumental_path,
        sample_rate=44100,
        frames=44100,
        elapsed_sec=1.25,
        ensemble_summary={"window_count": 2, "candidate_labels": ["mdx23c", "melband"]},
    )

    assert manifest["model_id"] == "reference_guided_vocal_ensemble"
    assert manifest["contract"]["benchmark_compatible"] is True
    assert manifest["contract"]["fallback_used"] is False
    assert manifest["contract"]["target_reference_assisted"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert manifest["reference_assistance"]["ground_truth_manifest"] == "/refs/ground_truth_manifest.json"
    assert manifest["stems"]["vocals"]["path"] == str(vocals_path)
    json.dumps(manifest)


def test_reference_guided_manifest_rejects_embedded_window_diagnostics(tmp_path: Path) -> None:
    module = _load_module()
    vocals_path = tmp_path / "vocals.flac"
    instrumental_path = tmp_path / "instrumental.flac"
    vocals_path.write_text("stub vocals", encoding="utf-8")
    instrumental_path.write_text("stub instrumental", encoding="utf-8")

    with pytest.raises(ValueError, match="per-window diagnostics"):
        module.build_reference_guided_manifest(
            source_audio_path=Path("/tracks/source.flac"),
            ground_truth_manifest=Path("/refs/ground_truth_manifest.json"),
            candidate_specs=[module.CandidateSpec(label="melband", manifest_path=Path("/runs/melband/stem_manifest.json"))],
            vocals_path=vocals_path,
            instrumental_path=instrumental_path,
            sample_rate=44100,
            frames=44100,
            elapsed_sec=1.25,
            ensemble_summary={
                "window_count": 2,
                "candidate_labels": ["melband"],
                "windows": [{"start_frame": 0, "end_frame": 100}],
            },
        )


def test_write_window_diagnostics_sidecar_keeps_manifest_compact(tmp_path: Path) -> None:
    module = _load_module()
    ensemble = module.EnsembleResult(
        audio=np.zeros((4, 2), dtype=np.float32),
        window_count=2,
        candidate_labels=["melband"],
        aggregate_weights=[1.0],
        window_summaries=[
            {"start_frame": 0, "end_frame": 2, "weights": {"melband": 1.0}},
            {"start_frame": 2, "end_frame": 4, "weights": {"melband": 1.0}},
        ],
    )

    sidecar = module.write_ensemble_window_diagnostics(
        output_dir=tmp_path,
        algorithm="test_algorithm",
        mode="time_domain",
        ensemble=ensemble,
        extra_summary={"band_edges_hz": [0.0, 22050.0]},
    )
    sidecar_path = Path(sidecar["path"])
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert sidecar_path.name == "ensemble_window_diagnostics.json"
    assert sidecar["window_count"] == 2
    assert sidecar["bytes"] == sidecar_path.stat().st_size
    assert sidecar["sha256"] == module.sha256_file(sidecar_path)
    assert payload["windows"] == ensemble.window_summaries

    vocals_path = tmp_path / "vocals.flac"
    instrumental_path = tmp_path / "instrumental.flac"
    vocals_path.write_text("stub vocals", encoding="utf-8")
    instrumental_path.write_text("stub instrumental", encoding="utf-8")
    manifest = module.build_reference_guided_manifest(
        source_audio_path=Path("/tracks/source.flac"),
        ground_truth_manifest=Path("/refs/ground_truth_manifest.json"),
        candidate_specs=[module.CandidateSpec(label="melband", manifest_path=Path("/runs/melband/stem_manifest.json"))],
        vocals_path=vocals_path,
        instrumental_path=instrumental_path,
        sample_rate=44100,
        frames=44100,
        elapsed_sec=1.25,
        ensemble_summary={
            "window_count": ensemble.window_count,
            "candidate_labels": ensemble.candidate_labels,
            "window_diagnostics": sidecar,
        },
    )

    assert "windows" not in manifest["ensemble"]
    assert manifest["ensemble"]["window_diagnostics"]["sha256"] == module.sha256_file(sidecar_path)
    assert manifest["ensemble"]["window_diagnostics"]["bytes"] == sidecar_path.stat().st_size


def test_candidate_validation_accepts_registered_execution_contract_when_stem_manifest_has_no_contract(
    tmp_path: Path,
) -> None:
    module = _load_module()
    manifest_path = tmp_path / "stem_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "htdemucs_ft",
                "stems": {"vocals": {"path": "vocals.flac"}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "registry_candidate_execution_manifest.json").write_text(
        json.dumps(
            {
                "contract": {
                    "fallback_used": False,
                    "registered_model_required": True,
                    "expected_manifest_required": True,
                    "hidden_inference": False,
                },
                "status": "completed",
                "candidate_manifest": {"exists": True, "path": str(manifest_path)},
            }
        ),
        encoding="utf-8",
    )

    module.validate_candidate_manifest(manifest_path)


def test_candidate_validation_accepts_relative_registered_candidate_manifest_path(
    tmp_path: Path,
) -> None:
    module = _load_module()
    manifest_path = tmp_path / "stem_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "htdemucs_ft",
                "stems": {"vocals": {"path": "vocals.flac"}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "registry_candidate_execution_manifest.json").write_text(
        json.dumps(
            {
                "contract": {
                    "fallback_used": False,
                    "registered_model_required": True,
                    "expected_manifest_required": True,
                    "hidden_inference": False,
                },
                "status": "completed",
                "candidate_manifest": {"exists": True, "path": "stem_manifest.json"},
            }
        ),
        encoding="utf-8",
    )

    module.validate_candidate_manifest(manifest_path)


def test_candidate_validation_rejects_legacy_execution_manifest_without_registered_contract(
    tmp_path: Path,
) -> None:
    module = _load_module()
    manifest_path = tmp_path / "stem_manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "model_id": "htdemucs_ft", "stems": {"vocals": {"path": "vocals.flac"}}}),
        encoding="utf-8",
    )
    (tmp_path / "registry_candidate_execution_manifest.json").write_text(
        json.dumps(
            {
                "contract": {
                    "fallback_used": False,
                    "expected_manifest_required": True,
                    "hidden_inference": False,
                },
                "status": "completed",
                "candidate_manifest": {"exists": True, "path": str(manifest_path)},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="registered_model_required"):
        module.validate_candidate_manifest(manifest_path)


def test_candidate_validation_rejects_legacy_execution_manifest_path_mismatch(
    tmp_path: Path,
) -> None:
    module = _load_module()
    manifest_path = tmp_path / "stem_manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "model_id": "htdemucs_ft", "stems": {"vocals": {"path": "vocals.flac"}}}),
        encoding="utf-8",
    )
    (tmp_path / "registry_candidate_execution_manifest.json").write_text(
        json.dumps(
            {
                "contract": {
                    "fallback_used": False,
                    "registered_model_required": True,
                    "expected_manifest_required": True,
                    "hidden_inference": False,
                },
                "status": "completed",
                "candidate_manifest": {"exists": True, "path": str(tmp_path / "other_stem_manifest.json")},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not match requested manifest"):
        module.validate_candidate_manifest(manifest_path)
