from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = TOOLS_ROOT / "build_reference_guided_stem_residual_repair.py"


def _load_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    spec = importlib.util.spec_from_file_location("build_reference_guided_stem_residual_repair", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiny_bundle(module, root: Path) -> tuple[Path, Path]:
    sample_rate = 128
    frames = 64
    source = np.zeros((frames, 2), dtype=np.float32)
    reference = np.ones((frames, 2), dtype=np.float32) * 0.25
    candidate = np.ones((frames, 2), dtype=np.float32) * 0.05
    source_path = root / "gt" / "source_mix.flac"
    reference_path = root / "gt" / "stems" / "drums.flac"
    candidate_path = root / "candidate" / "drums.flac"
    source_path.parent.mkdir(parents=True)
    reference_path.parent.mkdir(parents=True)
    candidate_path.parent.mkdir(parents=True)
    module.sf.write(str(source_path), source, sample_rate)
    module.sf.write(str(reference_path), reference, sample_rate)
    module.sf.write(str(candidate_path), candidate, sample_rate)

    ground_truth_manifest = root / "gt" / "ground_truth_manifest.json"
    ground_truth_manifest.write_text(
        json.dumps(
            {
                "ground_truth_contract": {
                    "verified": True,
                    "verification_method": "unit_test",
                    "source": "unit_test",
                },
                "mix": {"path": "source_mix.flac"},
                "stems": {"drums": {"path": "stems/drums.flac", "derived": False}},
            }
        ),
        encoding="utf-8",
    )
    candidate_manifest = root / "candidate" / "stem_manifest.json"
    candidate_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": {"fallback_used": False, "benchmark_compatible": True},
                "stems": {"drums": {"path": "drums.flac"}},
            }
        ),
        encoding="utf-8",
    )
    return ground_truth_manifest, candidate_manifest


def test_residual_repair_strength_one_matches_reference_when_gate_is_active() -> None:
    module = _load_module()
    candidate = np.ones((16, 2), dtype=np.float32) * 0.1
    reference = np.ones((16, 2), dtype=np.float32) * 0.5

    repaired, diagnostics = module.apply_reference_guided_residual_repair(
        candidate,
        reference,
        sample_rate=128,
        config=module.ResidualRepairConfig(residual_strength=1.0, envelope_window_ms=1.0),
    )

    assert np.allclose(repaired, reference, atol=1e-6)
    assert diagnostics["mean_gate"] > 0.99


def test_residual_repair_strength_zero_preserves_candidate() -> None:
    module = _load_module()
    candidate = np.ones((16, 2), dtype=np.float32) * 0.1
    reference = np.ones((16, 2), dtype=np.float32) * 0.5

    repaired, _ = module.apply_reference_guided_residual_repair(
        candidate,
        reference,
        sample_rate=128,
        config=module.ResidualRepairConfig(residual_strength=0.0, envelope_window_ms=1.0),
    )

    assert np.allclose(repaired, candidate, atol=1e-6)


def test_build_manifest_marks_reference_assisted_contract(tmp_path: Path) -> None:
    module = _load_module()
    ground_truth_manifest, candidate_manifest = _write_tiny_bundle(module, tmp_path)

    manifest = module.build_reference_guided_stem_residual_repair(
        ground_truth_manifest=ground_truth_manifest,
        candidate_manifest=candidate_manifest,
        output_dir=tmp_path / "out",
        stem_name="drums",
        config=module.ResidualRepairConfig(residual_strength=0.5, envelope_window_ms=1.0),
        listen_dir=tmp_path / "listen",
    )

    assert manifest["contract"]["target_reference_assisted"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert manifest["contract"]["fallback_used"] is False
    assert manifest["reference_assistance"]["not_a_no_reference_production_route"] is True
    assert sorted(manifest["stems"]) == ["drums"]
    assert (tmp_path / "out" / "drums.flac").exists()
    assert (tmp_path / "listen" / "02_reference_guided_residual_repair_drums.flac").exists()


def test_cli_requires_explicit_stem_to_keep_route_generic() -> None:
    module = _load_module()
    parser = module.build_arg_parser()

    try:
        parser.parse_args(
            [
                "--ground-truth-manifest",
                "ground_truth_manifest.json",
                "--candidate-manifest",
                "candidate_manifest.json",
                "--output-dir",
                "out",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("--stem must be required")
