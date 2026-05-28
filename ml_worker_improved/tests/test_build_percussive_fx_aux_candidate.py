import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_percussive_fx_aux_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_percussive_fx_aux_candidate", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_24")


def _safe_manifest(path: Path, stem_name: str, stem_path: Path, model_id: str) -> None:
    path.write_text(
        f"""{{
  "source_audio_path": "/app/input/source.flac",
  "model_id": "{model_id}",
  "contract": {{
    "benchmark_compatible": true,
    "target_reference_assisted": false,
    "target_reference_used_at_application": false,
    "application_reference_free": true,
    "delivery_allowed_without_reference": true,
    "fallback_used": false,
    "per_track_hardcode_allowed": false,
    "not_timestamp_hardcoded": true
  }},
  "stems": {{
    "{stem_name}": {{
      "path": "{stem_path.name}"
    }}
  }}
}}""",
        encoding="utf-8",
    )


def test_estimator_adds_transient_aux_without_copying_sustained_melody(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    frame_count = sample_rate * 2
    time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)

    base = np.zeros((frame_count, 2), dtype=np.float32)
    base[1000:1010] = 0.8

    clap_noise = np.zeros(frame_count, dtype=np.float32)
    rng = np.random.default_rng(7)
    clap_noise[5600:6000] = rng.normal(0.0, 0.35, 400).astype(np.float32)
    melody = 0.08 * np.sin(2.0 * np.pi * 440.0 * time)
    aux = np.column_stack([clap_noise + melody, clap_noise + melody]).astype(np.float32)

    base_dir = tmp_path / "base"
    aux_dir = tmp_path / "aux"
    base_audio = base_dir / "drums.flac"
    aux_audio = aux_dir / "other.flac"
    _write_audio(base_audio, base, sample_rate)
    _write_audio(aux_audio, aux, sample_rate)
    _safe_manifest(base_dir / "stem_manifest.json", "drums", base_audio, "strict_drum_candidate")
    _safe_manifest(aux_dir / "stem_manifest.json", "other", aux_audio, "full_stem_candidate")

    manifest = module.build_percussive_fx_aux_candidate(
        base_manifest_path=base_dir / "stem_manifest.json",
        aux_manifest_paths=[aux_dir / "stem_manifest.json"],
        output_dir=tmp_path / "out",
        route_name="test_percussive_fx_aux",
        config=module.PercussiveAuxConfig(
            frame_size=512,
            hop_size=128,
            min_flatness=0.18,
            flux_percentile=65.0,
            aux_gain=1.0,
            max_aux_rms_ratio=0.75,
        ),
    )

    rendered, _ = sf.read(manifest["stems"]["drums_plus_percussive_fx_aux"]["path"], always_2d=True, dtype="float32")
    residual = rendered - base
    clap_region = residual[5600:6000]
    melody_only_region = residual[2500:4500]
    clap_rms = float(np.sqrt(np.mean(clap_region * clap_region) + 1.0e-12))
    melody_rms = float(np.sqrt(np.mean(melody_only_region * melody_only_region) + 1.0e-12))

    assert manifest["contract"]["target_reference_assisted"] is False
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["pipeline"]["aux_estimator"]["source_stems_used"] == ["other"]
    assert "percussive_fx_aux" in manifest["stems"]
    assert manifest["stems"]["percussive_fx_aux"]["target_definition"] == "percussive_fx_aux"
    assert clap_rms > 0.04
    assert melody_rms < clap_rms * 0.25


def test_estimator_rejects_reference_assisted_aux_manifest(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    audio = np.zeros((sample_rate, 2), dtype=np.float32)
    base_dir = tmp_path / "base"
    aux_dir = tmp_path / "aux"
    base_audio = base_dir / "drums.flac"
    aux_audio = aux_dir / "other.flac"
    _write_audio(base_audio, audio, sample_rate)
    _write_audio(aux_audio, audio, sample_rate)
    _safe_manifest(base_dir / "stem_manifest.json", "drums", base_audio, "strict_drum_candidate")
    _safe_manifest(aux_dir / "stem_manifest.json", "other", aux_audio, "unsafe_aux_candidate")
    unsafe_manifest = (aux_dir / "stem_manifest.json").read_text(encoding="utf-8").replace(
        '"target_reference_used_at_application": false',
        '"target_reference_used_at_application": true',
    )
    (aux_dir / "stem_manifest.json").write_text(unsafe_manifest, encoding="utf-8")

    try:
        module.build_percussive_fx_aux_candidate(
            base_manifest_path=base_dir / "stem_manifest.json",
            aux_manifest_paths=[aux_dir / "stem_manifest.json"],
            output_dir=tmp_path / "out",
        )
    except ValueError as exc:
        assert "target reference at application time" in str(exc)
    else:
        raise AssertionError("unsafe aux manifest must be rejected")


def test_estimator_rejects_reference_used_aux_ancestor(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    audio = np.zeros((sample_rate, 2), dtype=np.float32)
    base_dir = tmp_path / "base"
    aux_dir = tmp_path / "aux"
    base_audio = base_dir / "drums.flac"
    aux_audio = aux_dir / "other.flac"
    _write_audio(base_audio, audio, sample_rate)
    _write_audio(aux_audio, audio, sample_rate)
    _safe_manifest(base_dir / "stem_manifest.json", "drums", base_audio, "strict_drum_candidate")
    _safe_manifest(aux_dir / "stem_manifest.json", "other", aux_audio, "full_stem_candidate")
    ancestor_path = aux_dir / "teacher_manifest.json"
    _safe_manifest(ancestor_path, "other", aux_audio, "reference_teacher")
    unsafe_ancestor = json.loads(ancestor_path.read_text(encoding="utf-8"))
    unsafe_ancestor["contract"]["target_reference_used_at_application"] = True
    unsafe_ancestor["contract"]["application_reference_free"] = False
    ancestor_path.write_text(json.dumps(unsafe_ancestor), encoding="utf-8")
    aux_manifest_path = aux_dir / "stem_manifest.json"
    aux_manifest = json.loads(aux_manifest_path.read_text(encoding="utf-8"))
    aux_manifest["pipeline"] = {"source_candidate_manifest": "teacher_manifest.json"}
    aux_manifest_path.write_text(json.dumps(aux_manifest), encoding="utf-8")

    try:
        module.build_percussive_fx_aux_candidate(
            base_manifest_path=base_dir / "stem_manifest.json",
            aux_manifest_paths=[aux_manifest_path],
            output_dir=tmp_path / "out",
        )
    except ValueError as exc:
        assert "target reference at application time" in str(exc)
    else:
        raise AssertionError("aux manifest with reference-used ancestor must be rejected")


def test_estimator_rejects_manifest_without_contract(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    audio = np.zeros((sample_rate, 2), dtype=np.float32)
    base_dir = tmp_path / "base"
    aux_dir = tmp_path / "aux"
    base_audio = base_dir / "drums.flac"
    aux_audio = aux_dir / "other.flac"
    _write_audio(base_audio, audio, sample_rate)
    _write_audio(aux_audio, audio, sample_rate)
    _safe_manifest(base_dir / "stem_manifest.json", "drums", base_audio, "strict_drum_candidate")
    _safe_manifest(aux_dir / "stem_manifest.json", "other", aux_audio, "full_stem_candidate")
    unsafe_manifest_path = aux_dir / "stem_manifest.json"
    unsafe_manifest = json.loads(unsafe_manifest_path.read_text(encoding="utf-8"))
    unsafe_manifest.pop("contract")
    unsafe_manifest_path.write_text(json.dumps(unsafe_manifest), encoding="utf-8")

    try:
        module.build_percussive_fx_aux_candidate(
            base_manifest_path=base_dir / "stem_manifest.json",
            aux_manifest_paths=[unsafe_manifest_path],
            output_dir=tmp_path / "out",
        )
    except ValueError as exc:
        assert "manifest has no contract" in str(exc)
    else:
        raise AssertionError("manifest without contract must be rejected")


def test_estimator_rejects_manifest_with_partial_contract(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    audio = np.zeros((sample_rate, 2), dtype=np.float32)
    base_dir = tmp_path / "base"
    aux_dir = tmp_path / "aux"
    base_audio = base_dir / "drums.flac"
    aux_audio = aux_dir / "other.flac"
    _write_audio(base_audio, audio, sample_rate)
    _write_audio(aux_audio, audio, sample_rate)
    _safe_manifest(base_dir / "stem_manifest.json", "drums", base_audio, "strict_drum_candidate")
    _safe_manifest(aux_dir / "stem_manifest.json", "other", aux_audio, "full_stem_candidate")
    unsafe_manifest_path = aux_dir / "stem_manifest.json"
    unsafe_manifest = json.loads(unsafe_manifest_path.read_text(encoding="utf-8"))
    unsafe_manifest["contract"].pop("application_reference_free")
    unsafe_manifest_path.write_text(json.dumps(unsafe_manifest), encoding="utf-8")

    try:
        module.build_percussive_fx_aux_candidate(
            base_manifest_path=base_dir / "stem_manifest.json",
            aux_manifest_paths=[unsafe_manifest_path],
            output_dir=tmp_path / "out",
        )
    except ValueError as exc:
        assert "contract is missing required key" in str(exc)
    else:
        raise AssertionError("manifest with partial contract must be rejected")


def test_estimator_rejects_base_manifest_without_contract(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    audio = np.zeros((sample_rate, 2), dtype=np.float32)
    base_dir = tmp_path / "base"
    aux_dir = tmp_path / "aux"
    base_audio = base_dir / "drums.flac"
    aux_audio = aux_dir / "other.flac"
    _write_audio(base_audio, audio, sample_rate)
    _write_audio(aux_audio, audio, sample_rate)
    _safe_manifest(base_dir / "stem_manifest.json", "drums", base_audio, "strict_drum_candidate")
    _safe_manifest(aux_dir / "stem_manifest.json", "other", aux_audio, "full_stem_candidate")
    unsafe_manifest_path = base_dir / "stem_manifest.json"
    unsafe_manifest = json.loads(unsafe_manifest_path.read_text(encoding="utf-8"))
    unsafe_manifest.pop("contract")
    unsafe_manifest_path.write_text(json.dumps(unsafe_manifest), encoding="utf-8")

    try:
        module.build_percussive_fx_aux_candidate(
            base_manifest_path=unsafe_manifest_path,
            aux_manifest_paths=[aux_dir / "stem_manifest.json"],
            output_dir=tmp_path / "out",
        )
    except ValueError as exc:
        assert "manifest has no contract" in str(exc)
    else:
        raise AssertionError("base manifest without contract must be rejected")


def test_estimator_rejects_base_manifest_with_partial_contract(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    audio = np.zeros((sample_rate, 2), dtype=np.float32)
    base_dir = tmp_path / "base"
    aux_dir = tmp_path / "aux"
    base_audio = base_dir / "drums.flac"
    aux_audio = aux_dir / "other.flac"
    _write_audio(base_audio, audio, sample_rate)
    _write_audio(aux_audio, audio, sample_rate)
    _safe_manifest(base_dir / "stem_manifest.json", "drums", base_audio, "strict_drum_candidate")
    _safe_manifest(aux_dir / "stem_manifest.json", "other", aux_audio, "full_stem_candidate")
    unsafe_manifest_path = base_dir / "stem_manifest.json"
    unsafe_manifest = json.loads(unsafe_manifest_path.read_text(encoding="utf-8"))
    unsafe_manifest["contract"].pop("application_reference_free")
    unsafe_manifest_path.write_text(json.dumps(unsafe_manifest), encoding="utf-8")

    try:
        module.build_percussive_fx_aux_candidate(
            base_manifest_path=unsafe_manifest_path,
            aux_manifest_paths=[aux_dir / "stem_manifest.json"],
            output_dir=tmp_path / "out",
        )
    except ValueError as exc:
        assert "contract is missing required key" in str(exc)
    else:
        raise AssertionError("base manifest with partial contract must be rejected")


def test_estimator_rejects_aux_manifests_without_requested_stems(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    audio = np.zeros((sample_rate, 2), dtype=np.float32)
    base_dir = tmp_path / "base"
    aux_dir = tmp_path / "aux"
    base_audio = base_dir / "drums.flac"
    aux_audio = aux_dir / "vocals.flac"
    _write_audio(base_audio, audio, sample_rate)
    _write_audio(aux_audio, audio, sample_rate)
    _safe_manifest(base_dir / "stem_manifest.json", "drums", base_audio, "strict_drum_candidate")
    _safe_manifest(aux_dir / "stem_manifest.json", "vocals", aux_audio, "full_stem_candidate")

    try:
        module.build_percussive_fx_aux_candidate(
            base_manifest_path=base_dir / "stem_manifest.json",
            aux_manifest_paths=[aux_dir / "stem_manifest.json"],
            output_dir=tmp_path / "out",
            source_stems=("other", "fx"),
        )
    except ValueError as exc:
        assert "no requested aux stems were found" in str(exc)
    else:
        raise AssertionError("aux manifests without requested stems must be rejected")


def test_listen_label_must_stay_inside_listen_dir(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    target_path = tmp_path / "out" / "drums_plus_percussive_fx_aux.flac"
    _write_audio(target_path, np.zeros((sample_rate, 2), dtype=np.float32), sample_rate)

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


def test_cli_does_not_run_inference(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    sample_rate = 8000
    audio = np.zeros((sample_rate, 2), dtype=np.float32)
    base_dir = tmp_path / "base"
    aux_dir = tmp_path / "aux"
    base_audio = base_dir / "drums.flac"
    aux_audio = aux_dir / "other.flac"
    _write_audio(base_audio, audio, sample_rate)
    _write_audio(aux_audio, audio, sample_rate)
    _safe_manifest(base_dir / "stem_manifest.json", "drums", base_audio, "strict_drum_candidate")
    _safe_manifest(aux_dir / "stem_manifest.json", "other", aux_audio, "full_stem_candidate")

    monkeypatch.setattr(module, "require_docker_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_percussive_fx_aux_candidate.py",
            "--base-manifest",
            str(base_dir / "stem_manifest.json"),
            "--aux-manifest",
            str(aux_dir / "stem_manifest.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--listen-dir",
            str(tmp_path / "listen"),
        ],
    )

    assert module.main() == 0
    assert (tmp_path / "out" / "stem_manifest.json").exists()
    assert (tmp_path / "listen" / "01_drums_plus_percussive_fx_aux.flac").exists()
    assert (tmp_path / "listen" / "02_percussive_fx_aux.flac").exists()
