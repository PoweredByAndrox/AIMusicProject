from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_percussive_event_dictionary_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_percussive_event_dictionary_candidate", MODULE_PATH)
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
        json.dumps(
            {
                "source_audio_path": None,
                "model_id": "tonal_stronger_v3_drums_plus_percussive_fx_aux",
                "contract": {
                    "target_reference_assisted": False,
                    "target_reference_used_at_application": False,
                    "application_reference_free": True,
                    "fallback_used": False,
                    "per_track_hardcode_allowed": False,
                    "not_timestamp_hardcoded": True,
                },
                "stems": {
                    "drums_plus_percussive_fx_aux": {
                        "path": stem_path.name,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_event_dictionary_adds_sparse_clap_without_bar_spanning_melody(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    frames = sample_rate * 3
    time = np.arange(frames, dtype=np.float32) / float(sample_rate)
    base = np.zeros((frames, 2), dtype=np.float32)
    base[1000:1024] = 0.72
    base[9000:9024] = -0.52

    melody = np.zeros(frames, dtype=np.float32)
    melody[3500:7200] = 0.12 * np.sin(2.0 * np.pi * 440.0 * time[3500:7200])
    rng = np.random.default_rng(23)
    clap = np.zeros(frames, dtype=np.float32)
    clap[13200:13640] = rng.normal(0.0, 0.36, 440).astype(np.float32) * np.linspace(1.0, 0.2, 440)
    source_mix = base + np.column_stack([melody + clap, melody + clap]).astype(np.float32)

    base_dir = tmp_path / "base"
    base_path = base_dir / "drums_plus_percussive_fx_aux.flac"
    source_path = tmp_path / "source_mix.flac"
    _write_audio(base_path, base, sample_rate)
    _write_audio(source_path, source_mix, sample_rate)
    _write_base_manifest(base_dir / "stem_manifest.json", base_path)

    manifest = module.build_percussive_event_dictionary_candidate(
        base_manifest_path=base_dir / "stem_manifest.json",
        source_audio_path=source_path,
        output_dir=tmp_path / "out",
        config=module.PercussiveEventDictionaryConfig(
            frame_size=512,
            hop_size=128,
            harmonic_time_bins=17,
            percussive_frequency_bins=17,
            mask_power=2.0,
            event_flux_percentile=58.0,
            min_event_flatness=0.08,
            pre_roll_frames=1,
            tail_frames=3,
            aux_gain=1.0,
            max_aux_rms_ratio=1.2,
        ),
    )

    rendered, _ = sf.read(manifest["stems"]["drums_plus_percussive_fx_aux"]["path"], always_2d=True, dtype="float32")
    residual = rendered - base
    clap_rms = float(np.sqrt(np.mean(residual[13200:13640] * residual[13200:13640]) + 1.0e-12))
    melody_rms = float(np.sqrt(np.mean(residual[4200:6900] * residual[4200:6900]) + 1.0e-12))

    np.testing.assert_allclose(rendered[1000:1024], base[1000:1024], atol=1.0e-5)
    assert clap_rms > 0.015
    assert melody_rms < clap_rms * 0.2
    assert manifest["contract"]["target_reference_assisted"] is False
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is True
    assert manifest["pipeline"]["family"] == "percussive_event_dictionary"
    assert manifest["pipeline"]["reference_audio_used"] is False
    assert manifest["pipeline"]["base_audio_unmodified"] is True


def test_event_dictionary_candidate_is_repeatable(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    frames = 4096
    base = np.zeros((frames, 2), dtype=np.float32)
    source_mix = base.copy()
    source_mix[1400:1450] = 0.25
    base_dir = tmp_path / "base"
    base_path = base_dir / "drums_plus_percussive_fx_aux.flac"
    source_path = tmp_path / "source_mix.flac"
    _write_audio(base_path, base, sample_rate)
    _write_audio(source_path, source_mix, sample_rate)
    _write_base_manifest(base_dir / "stem_manifest.json", base_path)

    first = module.build_percussive_event_dictionary_candidate(
        base_manifest_path=base_dir / "stem_manifest.json",
        source_audio_path=source_path,
        output_dir=tmp_path / "out_a",
    )
    second = module.build_percussive_event_dictionary_candidate(
        base_manifest_path=base_dir / "stem_manifest.json",
        source_audio_path=source_path,
        output_dir=tmp_path / "out_b",
    )

    first_audio, _ = sf.read(first["stems"]["drums_plus_percussive_fx_aux"]["path"], always_2d=True, dtype="float32")
    second_audio, _ = sf.read(second["stems"]["drums_plus_percussive_fx_aux"]["path"], always_2d=True, dtype="float32")
    np.testing.assert_allclose(first_audio, second_audio, atol=0.0)
    assert first["pipeline"]["deterministic"] is True
    assert second["pipeline"]["deterministic"] is True


def test_fit_frame_gate_pads_missing_frames_as_inactive() -> None:
    module = _load_module()

    padded = module._fit_frame_gate(np.array([False, True], dtype=bool), 5)

    np.testing.assert_array_equal(padded, np.array([False, True, False, False, False], dtype=bool))


def test_aux_sanity_rejects_broad_harmonic_residual() -> None:
    module = _load_module()
    sample_rate = 8000
    frames = sample_rate * 3
    time = np.arange(frames, dtype=np.float32) / float(sample_rate)
    harmonic_residual = np.zeros((frames, 2), dtype=np.float32)
    harmonic_residual[: sample_rate * 2, 0] = 0.05 * np.sin(2.0 * np.pi * 440.0 * time[: sample_rate * 2])
    harmonic_residual[: sample_rate * 2, 1] = harmonic_residual[: sample_rate * 2, 0]

    report = module.evaluate_percussive_aux_sanity(
        harmonic_residual,
        sample_rate=sample_rate,
        config=module.PercussiveEventDictionaryConfig(),
    )

    assert report["status"] == "rejected"
    assert "aux_not_sparse" in report["reject_reasons"]
    assert "harmonic_melody_dominant" in report["reject_reasons"]


def test_cli_blocks_listen_pack_when_aux_sanity_rejects(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    target_path = tmp_path / "out" / "drums_plus_percussive_fx_aux.flac"
    aux_path = tmp_path / "out" / "estimated_percussive_event_aux.flac"
    manifest_path = tmp_path / "out" / "stem_manifest.json"
    target_path.parent.mkdir(parents=True)
    target_path.write_bytes(b"target")
    aux_path.write_bytes(b"bad-aux")

    def fake_build_candidate(**_kwargs):
        manifest = {
            "manifest_path": str(manifest_path),
            "pipeline": {
                "aux_sanity": {
                    "status": "rejected",
                    "reject_reasons": ["aux_not_sparse", "harmonic_melody_dominant"],
                }
            },
            "stems": {
                module.TARGET_STEM: {"path": str(target_path)},
                "percussive_fx_aux": {"path": str(aux_path)},
            },
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    listen_dir = tmp_path / "listen"
    monkeypatch.setattr(module, "require_docker_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(module, "build_percussive_event_dictionary_candidate", fake_build_candidate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_percussive_event_dictionary_candidate.py",
            "--base-manifest",
            str(tmp_path / "base_manifest.json"),
            "--source-audio",
            str(tmp_path / "source.flac"),
            "--output-dir",
            str(tmp_path / "out"),
            "--listen-dir",
            str(listen_dir),
            "--listen-label",
            "candidate.flac",
        ],
    )

    assert module.main() == 0
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest["listen"]["status"] == "blocked"
    assert saved_manifest["listen"]["reason"] == "percussive_event_aux_sanity_failed"
    assert not (listen_dir / "candidate.flac").exists()
    assert not (listen_dir / "02_percussive_event_aux.flac").exists()


def test_rejected_aux_sanity_blocks_no_reference_delivery_contract(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    sample_rate = 8000
    frames = 2048
    base = np.zeros((frames, 2), dtype=np.float32)
    source_mix = base.copy()
    base_dir = tmp_path / "base"
    base_path = base_dir / "drums_plus_percussive_fx_aux.flac"
    source_path = tmp_path / "source_mix.flac"
    _write_audio(base_path, base, sample_rate)
    _write_audio(source_path, source_mix, sample_rate)
    _write_base_manifest(base_dir / "stem_manifest.json", base_path)

    monkeypatch.setattr(
        module,
        "evaluate_percussive_aux_sanity",
        lambda *_args, **_kwargs: {
            "status": "rejected",
            "reject_reasons": ["aux_not_sparse", "harmonic_melody_dominant"],
        },
    )

    manifest = module.build_percussive_event_dictionary_candidate(
        base_manifest_path=base_dir / "stem_manifest.json",
        source_audio_path=source_path,
        output_dir=tmp_path / "out",
    )

    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert manifest["contract"]["no_reference_production_claim"] is False
    assert manifest["contract"]["delivery_blocked_reason"] == "percussive_event_aux_sanity_failed"
    assert manifest["delivery"]["status"] == "rejected"
    assert manifest["delivery"]["reject_reasons"] == ["aux_not_sparse", "harmonic_melody_dominant"]
