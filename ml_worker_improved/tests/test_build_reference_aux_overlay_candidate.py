from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_reference_aux_overlay_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_reference_aux_overlay_candidate", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_24")


def test_overlay_preserves_candidate_and_adds_reference_aux_delta(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    frames = 4000
    candidate = np.zeros((frames, 2), dtype=np.float32)
    candidate[1000:1010] = 0.7
    reference_drums = np.zeros((frames, 2), dtype=np.float32)
    reference_fx = np.zeros((frames, 2), dtype=np.float32)
    reference_fx[2200:2400, 0] = 0.25
    reference_fx[2200:2400, 1] = -0.2
    reference_target = reference_drums + reference_fx

    candidate_dir = tmp_path / "candidate"
    gt_dir = tmp_path / "gt"
    candidate_path = candidate_dir / "drums_plus_percussive_fx_aux.flac"
    reference_drums_path = gt_dir / "stems" / "drums.flac"
    reference_target_path = gt_dir / "stems" / "drums_plus_percussive_fx_aux.flac"
    _write_audio(candidate_path, candidate, sample_rate)
    _write_audio(reference_drums_path, reference_drums, sample_rate)
    _write_audio(reference_target_path, reference_target, sample_rate)
    candidate_manifest_path = candidate_dir / "stem_manifest.json"
    candidate_manifest_path.write_text(
        """{
  "source_audio_path": "/app/input/source.flac",
  "model_id": "tonal_stronger_v3_drums_plus_percussive_fx_aux",
  "contract": {
    "target_reference_assisted": false,
    "target_reference_used_at_application": false,
    "application_reference_free": true,
    "fallback_used": false,
    "per_track_hardcode_allowed": false,
    "not_timestamp_hardcoded": true
  },
  "stems": {
    "drums_plus_percussive_fx_aux": {
      "path": "drums_plus_percussive_fx_aux.flac"
    }
  }
}""",
        encoding="utf-8",
    )
    ground_truth_manifest_path = gt_dir / "ground_truth_manifest.json"
    ground_truth_manifest_path.write_text(
        """{
  "mix": {"path": "source_mix.flac"},
  "stems": {
    "drums": {"path": "stems/drums.flac"},
    "drums_plus_percussive_fx_aux": {"path": "stems/drums_plus_percussive_fx_aux.flac"}
  }
}""",
        encoding="utf-8",
    )

    manifest = module.build_reference_aux_overlay_candidate(
        candidate_manifest_path=candidate_manifest_path,
        ground_truth_manifest_path=ground_truth_manifest_path,
        output_dir=tmp_path / "out",
    )

    rendered, _ = sf.read(manifest["stems"]["drums_plus_percussive_fx_aux"]["path"], always_2d=True, dtype="float32")
    np.testing.assert_allclose(rendered[1000:1010], candidate[1000:1010], atol=1.0e-5)
    np.testing.assert_allclose(rendered[2200:2400], reference_fx[2200:2400], atol=1.0e-5)
    assert manifest["contract"]["target_reference_assisted"] is True
    assert manifest["contract"]["target_reference_used_at_application"] is True
    assert manifest["contract"]["application_reference_free"] is False
    assert manifest["contract"]["no_reference_production_claim"] is False
    assert manifest["pipeline"]["base_preserving_addition"] is True


def test_overlay_aux_gain_controls_added_delta(tmp_path: Path) -> None:
    module = _load_module()
    candidate = np.zeros((2, 2), dtype=np.float32)
    aux = np.array([[0.4, -0.2], [0.2, 0.1]], dtype=np.float32)

    rendered, diagnostics = module.add_aux_overlay_preserving_base(candidate, aux, aux_gain=0.5, max_peak=0.99)

    np.testing.assert_allclose(rendered, aux * 0.5, atol=1.0e-6)
    assert diagnostics["aux_gain"] == 0.5
    assert diagnostics["base_audio_modified"] is False


def test_overlay_attenuates_aux_only_when_peak_headroom_is_exceeded(tmp_path: Path) -> None:
    module = _load_module()
    candidate = np.array([[0.95, -0.95], [0.1, 0.1]], dtype=np.float32)
    aux = np.array([[0.2, -0.2], [0.2, 0.2]], dtype=np.float32)

    rendered, diagnostics = module.add_aux_overlay_preserving_base(candidate, aux, aux_gain=1.0, max_peak=0.99)

    np.testing.assert_allclose(rendered[0], np.array([0.99, -0.99], dtype=np.float32), atol=1.0e-6)
    np.testing.assert_allclose(rendered[1], np.array([0.3, 0.3], dtype=np.float32), atol=1.0e-6)
    assert diagnostics["peak_limited_by_aux_attenuation"] is True
    assert diagnostics["base_audio_modified"] is False


def test_cli_does_not_claim_no_reference_delivery(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    sample_rate = 8000
    audio = np.zeros((sample_rate, 2), dtype=np.float32)
    candidate_dir = tmp_path / "candidate"
    gt_dir = tmp_path / "gt"
    _write_audio(candidate_dir / "drums_plus_percussive_fx_aux.flac", audio, sample_rate)
    _write_audio(gt_dir / "stems" / "drums.flac", audio, sample_rate)
    _write_audio(gt_dir / "stems" / "drums_plus_percussive_fx_aux.flac", audio, sample_rate)
    (candidate_dir / "stem_manifest.json").write_text(
        """{
  "source_audio_path": "/app/input/source.flac",
  "model_id": "safe_base",
  "contract": {
    "target_reference_assisted": false,
    "target_reference_used_at_application": false,
    "application_reference_free": true,
    "fallback_used": false,
    "per_track_hardcode_allowed": false,
    "not_timestamp_hardcoded": true
  },
  "stems": {"drums_plus_percussive_fx_aux": {"path": "drums_plus_percussive_fx_aux.flac"}}
}""",
        encoding="utf-8",
    )
    (gt_dir / "ground_truth_manifest.json").write_text(
        """{
  "mix": {"path": "source_mix.flac"},
  "stems": {
    "drums": {"path": "stems/drums.flac"},
    "drums_plus_percussive_fx_aux": {"path": "stems/drums_plus_percussive_fx_aux.flac"}
  }
}""",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "require_docker_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_reference_aux_overlay_candidate.py",
            "--candidate-manifest",
            str(candidate_dir / "stem_manifest.json"),
            "--ground-truth-manifest",
            str(gt_dir / "ground_truth_manifest.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--listen-dir",
            str(tmp_path / "listen"),
        ],
    )

    assert module.main() == 0
    manifest = module.read_json(tmp_path / "out" / "stem_manifest.json")
    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert (tmp_path / "listen" / "03_reference_aux_overlay_drums_plus_percussive_fx_aux.flac").exists()
