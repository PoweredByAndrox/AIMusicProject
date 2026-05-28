from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = TOOLS_ROOT / "build_reference_teacher_stem_pack.py"


def _load_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    spec = importlib.util.spec_from_file_location("build_reference_teacher_stem_pack", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiny_ground_truth_bundle(module, root: Path) -> Path:
    sample_rate = 128
    frames = 32
    audio = np.zeros((frames, 2), dtype=np.float32)
    source_path = root / "source_mix.flac"
    stem_dir = root / "stems"
    stem_dir.mkdir(parents=True)
    module.sf.write(str(source_path), audio, sample_rate)
    for stem_name in ("vocals", "drums", "instrumental", "drums_plus_percussive_fx_aux"):
        module.sf.write(str(stem_dir / f"{stem_name}.flac"), audio, sample_rate)
    manifest_path = root / "ground_truth_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "ground_truth_contract": {
                    "verified": True,
                    "verification_method": "unit_test",
                    "source": "unit_test",
                },
                "mix": {"path": "source_mix.flac"},
                "stems": {
                    "vocals": {"path": "stems/vocals.flac", "derived": False},
                    "drums": {"path": "stems/drums.flac", "derived": False},
                    "instrumental": {"path": "stems/instrumental.flac", "derived": True},
                    "drums_plus_percussive_fx_aux": {
                        "path": "stems/drums_plus_percussive_fx_aux.flac",
                        "derived": True,
                    },
                },
                "target_definitions": {
                    "drums_plus_percussive_fx_aux": {
                        "derived": True,
                        "child_stems": ["drums", "fx"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_teacher_pack_defaults_to_atomic_stems_only(tmp_path: Path) -> None:
    module = _load_module()
    manifest_path = _write_tiny_ground_truth_bundle(module, tmp_path / "gt")

    manifest = module.build_reference_teacher_stem_pack(
        ground_truth_manifest=manifest_path,
        output_dir=tmp_path / "out",
        requested_stems=[],
        include_derived=False,
        listen_dir=None,
    )

    assert sorted(manifest["stems"]) == ["drums", "vocals"]
    assert manifest["contract"]["target_reference_assisted"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert manifest["reference_assistance"]["not_a_no_reference_production_route"] is True
    assert (tmp_path / "out" / "stems" / "vocals.flac").exists()
    assert (tmp_path / "out" / "stems" / "drums.flac").exists()
    assert not (tmp_path / "out" / "stems" / "instrumental.flac").exists()


def test_teacher_pack_can_include_derived_instrumental(tmp_path: Path) -> None:
    module = _load_module()
    manifest_path = _write_tiny_ground_truth_bundle(module, tmp_path / "gt")

    manifest = module.build_reference_teacher_stem_pack(
        ground_truth_manifest=manifest_path,
        output_dir=tmp_path / "out",
        requested_stems=[],
        include_derived=True,
        listen_dir=tmp_path / "listen",
    )

    assert sorted(manifest["stems"]) == ["drums", "drums_plus_percussive_fx_aux", "instrumental", "vocals"]
    assert manifest["stems"]["instrumental"]["derived"] is True
    assert manifest["stems"]["drums_plus_percussive_fx_aux"]["derived"] is True
    assert (tmp_path / "listen" / "02_drums_plus_percussive_fx_aux_reference_teacher.flac").exists()
    assert (tmp_path / "listen" / "03_instrumental_reference_teacher.flac").exists()
    assert (tmp_path / "listen" / "candidate_map.json").exists()


def test_teacher_pack_rejects_missing_requested_stem(tmp_path: Path) -> None:
    module = _load_module()
    manifest_path = _write_tiny_ground_truth_bundle(module, tmp_path / "gt")

    try:
        module.build_reference_teacher_stem_pack(
            ground_truth_manifest=manifest_path,
            output_dir=tmp_path / "out",
            requested_stems=["piano"],
            include_derived=False,
            listen_dir=None,
        )
    except RuntimeError as exc:
        assert "Requested stem is missing" in str(exc)
    else:
        raise AssertionError("missing requested stem must fail")


def test_teacher_pack_requires_include_derived_for_requested_composite_target(tmp_path: Path) -> None:
    module = _load_module()
    manifest_path = _write_tiny_ground_truth_bundle(module, tmp_path / "gt")

    try:
        module.build_reference_teacher_stem_pack(
            ground_truth_manifest=manifest_path,
            output_dir=tmp_path / "out",
            requested_stems=["drums_plus_percussive_fx_aux"],
            include_derived=False,
            listen_dir=None,
        )
    except RuntimeError as exc:
        assert "Requested stem is derived" in str(exc)
    else:
        raise AssertionError("requested composite target must require --include-derived")
