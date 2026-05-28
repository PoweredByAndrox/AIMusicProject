from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_fullmix_residual_percussive_aux_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_fullmix_residual_percussive_aux_candidate", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_24")


def _write_base_manifest(path: Path, stem_path: Path) -> None:
    path.write_text(
        f"""{{
  "source_audio_path": null,
  "model_id": "tonal_stronger_v3_drums_plus_percussive_fx_aux",
  "contract": {{
    "target_reference_assisted": false,
    "target_reference_used_at_application": false,
    "application_reference_free": true,
    "fallback_used": false,
    "per_track_hardcode_allowed": false,
    "not_timestamp_hardcoded": true
  }},
  "stems": {{
    "drums_plus_percussive_fx_aux": {{
      "path": "{stem_path.name}"
    }}
  }}
}}""",
        encoding="utf-8",
    )


def test_fullmix_residual_route_preserves_base_and_extracts_sparse_clap_not_sustained_melody(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    frames = sample_rate * 2
    time = np.arange(frames, dtype=np.float32) / float(sample_rate)
    base = np.zeros((frames, 2), dtype=np.float32)
    base[1200:1210] = 0.75
    clap = np.zeros(frames, dtype=np.float32)
    rng = np.random.default_rng(11)
    clap[6000:6400] = rng.normal(0.0, 0.35, 400).astype(np.float32)
    melody = 0.08 * np.sin(2.0 * np.pi * 440.0 * time)
    full_mix = base + np.column_stack([clap + melody, clap + melody]).astype(np.float32)

    base_dir = tmp_path / "base"
    base_path = base_dir / "drums_plus_percussive_fx_aux.flac"
    source_path = tmp_path / "source_mix.flac"
    _write_audio(base_path, base, sample_rate)
    _write_audio(source_path, full_mix, sample_rate)
    _write_base_manifest(base_dir / "stem_manifest.json", base_path)

    manifest = module.build_fullmix_residual_percussive_aux_candidate(
        base_manifest_path=base_dir / "stem_manifest.json",
        source_audio_path=source_path,
        output_dir=tmp_path / "out",
        config=module.ResidualPercussiveAuxConfig(
            frame_size=512,
            hop_size=128,
            min_flatness=0.18,
            flux_percentile=65.0,
            aux_gain=1.0,
            max_aux_rms_ratio=0.75,
            limiter_peak=0.999,
        ),
    )

    rendered, _ = sf.read(manifest["stems"]["drums_plus_percussive_fx_aux"]["path"], always_2d=True, dtype="float32")
    residual = rendered - base
    clap_rms = float(np.sqrt(np.mean(residual[6000:6400] * residual[6000:6400]) + 1.0e-12))
    melody_rms = float(np.sqrt(np.mean(residual[2500:4500] * residual[2500:4500]) + 1.0e-12))
    np.testing.assert_allclose(rendered[1200:1210], base[1200:1210], atol=1.0e-5)
    assert clap_rms > 0.04
    assert melody_rms < clap_rms * 0.25
    assert manifest["contract"]["target_reference_assisted"] is False
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert "percussive_fx_aux" in manifest["stems"]
    assert manifest["stems"]["percussive_fx_aux"]["source_audio_role"] == "full_mix_minus_base_residual"
    assert manifest["pipeline"]["base_preserving_addition"] is True
    assert manifest["pipeline"]["source_audio_role"] == "full_mix"


def test_fullmix_residual_route_is_repeatable_for_same_inputs(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    frames = 4096
    base = np.zeros((frames, 2), dtype=np.float32)
    base[100:110] = 0.4
    full_mix = base.copy()
    full_mix[1000:1020] += 0.2
    base_dir = tmp_path / "base"
    base_path = base_dir / "drums_plus_percussive_fx_aux.flac"
    source_path = tmp_path / "source_mix.flac"
    _write_audio(base_path, base, sample_rate)
    _write_audio(source_path, full_mix, sample_rate)
    _write_base_manifest(base_dir / "stem_manifest.json", base_path)

    first = module.build_fullmix_residual_percussive_aux_candidate(
        base_manifest_path=base_dir / "stem_manifest.json",
        source_audio_path=source_path,
        output_dir=tmp_path / "out_a",
    )
    second = module.build_fullmix_residual_percussive_aux_candidate(
        base_manifest_path=base_dir / "stem_manifest.json",
        source_audio_path=source_path,
        output_dir=tmp_path / "out_b",
    )

    first_audio, _ = sf.read(first["stems"]["drums_plus_percussive_fx_aux"]["path"], always_2d=True, dtype="float32")
    second_audio, _ = sf.read(second["stems"]["drums_plus_percussive_fx_aux"]["path"], always_2d=True, dtype="float32")
    np.testing.assert_allclose(first_audio, second_audio, atol=0.0)
    assert first["pipeline"]["deterministic"] is True
    assert second["pipeline"]["deterministic"] is True


def test_cli_does_not_run_reference_assisted_route(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    sample_rate = 8000
    audio = np.zeros((sample_rate, 2), dtype=np.float32)
    base_dir = tmp_path / "base"
    base_path = base_dir / "drums_plus_percussive_fx_aux.flac"
    source_path = tmp_path / "source_mix.flac"
    _write_audio(base_path, audio, sample_rate)
    _write_audio(source_path, audio, sample_rate)
    _write_base_manifest(base_dir / "stem_manifest.json", base_path)
    monkeypatch.setattr(module, "require_docker_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_fullmix_residual_percussive_aux_candidate.py",
            "--base-manifest",
            str(base_dir / "stem_manifest.json"),
            "--source-audio",
            str(source_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--listen-dir",
            str(tmp_path / "listen"),
        ],
    )

    assert module.main() == 0
    manifest = module.read_json(tmp_path / "out" / "stem_manifest.json")
    assert manifest["contract"]["delivery_allowed_without_reference"] is True
    assert manifest["contract"]["no_reference_production_claim"] is True
    assert manifest["pipeline"]["reference_audio_used"] is False
    assert (tmp_path / "listen" / "01_fullmix_residual_percussive_aux.flac").exists()
    assert (tmp_path / "listen" / "02_percussive_fx_aux.flac").exists()


def test_fullmix_listen_label_must_stay_inside_listen_dir(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    target_path = tmp_path / "out" / "drums_plus_percussive_fx_aux.flac"
    aux_path = tmp_path / "out" / "estimated_fullmix_residual_percussive_aux.flac"
    _write_audio(target_path, np.zeros((sample_rate, 2), dtype=np.float32), sample_rate)
    _write_audio(aux_path, np.zeros((sample_rate, 2), dtype=np.float32), sample_rate)

    try:
        module._copy_listen_file(
            target_path=target_path,
            listen_dir=tmp_path / "listen",
            listen_label="../escaped.flac",
        )
    except ValueError as exc:
        assert "listen label must stay inside listen_dir" in str(exc)
    else:
        raise AssertionError("path-traversing listen labels must be rejected")


def test_fullmix_residual_route_rejects_reference_used_base_ancestor(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    audio = np.zeros((sample_rate, 2), dtype=np.float32)
    base_dir = tmp_path / "base"
    base_path = base_dir / "drums_plus_percussive_fx_aux.flac"
    source_path = tmp_path / "source_mix.flac"
    _write_audio(base_path, audio, sample_rate)
    _write_audio(source_path, audio, sample_rate)
    _write_base_manifest(base_dir / "stem_manifest.json", base_path)
    ancestor_path = base_dir / "teacher_manifest.json"
    _write_base_manifest(ancestor_path, base_path)
    unsafe_ancestor = module.read_json(ancestor_path)
    unsafe_ancestor["contract"]["target_reference_used_at_application"] = True
    unsafe_ancestor["contract"]["application_reference_free"] = False
    ancestor_path.write_text(json.dumps(unsafe_ancestor), encoding="utf-8")
    base_manifest_path = base_dir / "stem_manifest.json"
    base_manifest = module.read_json(base_manifest_path)
    base_manifest["pipeline"] = {"base_export_manifest": "teacher_manifest.json"}
    base_manifest_path.write_text(json.dumps(base_manifest), encoding="utf-8")

    try:
        module.build_fullmix_residual_percussive_aux_candidate(
            base_manifest_path=base_manifest_path,
            source_audio_path=source_path,
            output_dir=tmp_path / "out",
        )
    except ValueError as exc:
        assert "target reference at application time" in str(exc)
    else:
        raise AssertionError("fullmix residual route must reject reference-used base ancestors")
