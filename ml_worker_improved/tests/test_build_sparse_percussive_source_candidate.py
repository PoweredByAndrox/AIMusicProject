from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_sparse_percussive_source_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_sparse_percussive_source_candidate", MODULE_PATH)
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
                        "path": str(stem_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_sparse_source_route_extracts_clap_without_sustained_melody(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    frames = sample_rate * 3
    time = np.arange(frames, dtype=np.float32) / float(sample_rate)
    base = np.zeros((frames, 2), dtype=np.float32)
    base[1000:1024] = 0.65
    base[9000:9024] = -0.45

    melody = np.zeros(frames, dtype=np.float32)
    melody[2400:12000] = 0.12 * np.sin(2.0 * np.pi * 440.0 * time[2400:12000])
    rng = np.random.default_rng(37)
    clap = np.zeros(frames, dtype=np.float32)
    clap[15000:15380] = rng.normal(0.0, 0.35, 380).astype(np.float32) * np.linspace(1.0, 0.15, 380)
    source_mix = base + np.column_stack([melody + clap, melody + clap]).astype(np.float32)

    base_dir = tmp_path / "base"
    base_path = base_dir / "drums_plus_percussive_fx_aux.flac"
    source_path = tmp_path / "source_mix.flac"
    _write_audio(base_path, base, sample_rate)
    _write_audio(source_path, source_mix, sample_rate)
    _write_base_manifest(base_dir / "stem_manifest.json", base_path)

    manifest = module.build_sparse_percussive_source_candidate(
        base_manifest_path=base_dir / "stem_manifest.json",
        source_audio_path=source_path,
        output_dir=tmp_path / "out",
        config=module.SparsePercussiveSourceConfig(
            frame_size=512,
            hop_size=128,
            source_flux_percentile=58.0,
            min_event_flatness=0.08,
            min_high_band_ratio=0.08,
            max_raw_event_frames=8,
            pre_roll_frames=1,
            tail_frames=2,
            aux_gain=1.0,
            max_aux_rms_ratio=0.75,
            max_aux_active_fraction=0.22,
            max_aux_low_mid_to_high_ratio=2.2,
        ),
    )

    rendered, _ = sf.read(manifest["stems"]["drums_plus_percussive_fx_aux"]["path"], always_2d=True, dtype="float32")
    residual = rendered - base
    clap_rms = float(np.sqrt(np.mean(residual[15000:15380] * residual[15000:15380]) + 1.0e-12))
    melody_rms = float(np.sqrt(np.mean(residual[4000:11000] * residual[4000:11000]) + 1.0e-12))

    np.testing.assert_allclose(rendered[1000:1024], base[1000:1024], atol=1.0e-5)
    assert clap_rms > 0.015
    assert melody_rms < clap_rms * 0.2
    assert manifest["delivery"]["status"] == "requires_objective_quality_gate"
    assert manifest["delivery"]["reason"] == "objective_quality_gate_required"
    assert manifest["contract"]["delivery_allowed_without_reference"] is True
    assert manifest["contract"]["fallback_used"] is False
    assert manifest["contract"]["per_track_hardcode_allowed"] is False
    assert manifest["contract"]["not_timestamp_hardcoded"] is True
    assert manifest["contract"]["no_reference_production_claim"] is False
    assert manifest["presentation_status"] == "diagnostic_requires_objective_gate"
    assert manifest["ralph_ai_delivery_contract"]["status"] == "requires_objective_gate"
    assert manifest["ralph_ai_delivery_contract"]["result_delivery_allowed"] is False
    assert manifest["quality_gate"]["status"] == "required"
    assert manifest["pipeline"]["source_event_metrics"]["kept_event_count"] >= 1
    assert manifest["pipeline"]["source_audio_role"] == "full_mix_sparse_percussive_source"


def test_sparse_source_route_blocks_harmonic_only_listen_pack(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    sample_rate = 8000
    frames = sample_rate * 3
    time = np.arange(frames, dtype=np.float32) / float(sample_rate)
    base = np.zeros((frames, 2), dtype=np.float32)
    harmonic = 0.06 * np.sin(2.0 * np.pi * 440.0 * time)
    source_mix = base + np.column_stack([harmonic, harmonic]).astype(np.float32)

    base_dir = tmp_path / "base"
    base_path = base_dir / "drums_plus_percussive_fx_aux.flac"
    source_path = tmp_path / "source_mix.flac"
    listen_dir = tmp_path / "listen"
    _write_audio(base_path, base, sample_rate)
    _write_audio(source_path, source_mix, sample_rate)
    _write_base_manifest(base_dir / "stem_manifest.json", base_path)
    monkeypatch.setattr(module, "require_docker_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_sparse_percussive_source_candidate.py",
            "--base-manifest",
            str(base_dir / "stem_manifest.json"),
            "--source-audio",
            str(source_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--listen-dir",
            str(listen_dir),
            "--listen-label",
            "candidate.flac",
            "--frame-size",
            "512",
            "--hop-size",
            "128",
        ],
    )

    assert module.main() == 0
    manifest = json.loads((tmp_path / "out" / "stem_manifest.json").read_text(encoding="utf-8"))
    assert manifest["delivery"]["status"] == "rejected"
    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert manifest["contract"]["no_reference_production_claim"] is False
    assert manifest["presentation_status"] == "blocked_from_delivery"
    assert manifest["ralph_ai_delivery_contract"]["status"] == "blocked"
    assert manifest["ralph_ai_delivery_contract"]["result_delivery_allowed"] is False
    assert manifest["ralph_ai_delivery_contract"]["rejects_best_of_bad_candidate"] is True
    assert manifest["listen"]["status"] == "blocked"
    assert not (listen_dir / "candidate.flac").exists()
    assert not (listen_dir / "02_sparse_percussive_source_aux.flac").exists()


def test_sparse_source_route_publishes_only_diagnostic_listen_before_objective_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    sample_rate = 8000
    frames = sample_rate * 3
    base = np.zeros((frames, 2), dtype=np.float32)
    rng = np.random.default_rng(41)
    clap = np.zeros(frames, dtype=np.float32)
    clap[15000:15380] = rng.normal(0.0, 0.35, 380).astype(np.float32) * np.linspace(1.0, 0.15, 380)
    source_mix = base + np.column_stack([clap, clap]).astype(np.float32)

    base_dir = tmp_path / "base"
    base_path = base_dir / "drums_plus_percussive_fx_aux.flac"
    source_path = tmp_path / "source_mix.flac"
    listen_dir = tmp_path / "listen"
    _write_audio(base_path, base, sample_rate)
    _write_audio(source_path, source_mix, sample_rate)
    _write_base_manifest(base_dir / "stem_manifest.json", base_path)
    monkeypatch.setattr(module, "require_docker_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_sparse_percussive_source_candidate.py",
            "--base-manifest",
            str(base_dir / "stem_manifest.json"),
            "--source-audio",
            str(source_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--listen-dir",
            str(listen_dir),
            "--listen-label",
            "candidate.flac",
            "--frame-size",
            "512",
            "--hop-size",
            "128",
            "--source-flux-percentile",
            "58",
            "--min-event-flatness",
            "0.08",
            "--min-high-band-ratio",
            "0.08",
            "--max-raw-event-frames",
            "8",
            "--pre-roll-frames",
            "1",
            "--tail-frames",
            "2",
            "--aux-gain",
            "1.0",
            "--max-aux-rms-ratio",
            "0.75",
            "--max-aux-active-fraction",
            "0.22",
            "--max-aux-low-mid-to-high-ratio",
            "2.2",
        ],
    )

    assert module.main() == 0
    manifest = json.loads((tmp_path / "out" / "stem_manifest.json").read_text(encoding="utf-8"))

    assert manifest["delivery"]["status"] == "requires_objective_quality_gate"
    assert manifest["delivery"]["reason"] == "objective_quality_gate_required"
    assert manifest["presentation_status"] == "diagnostic_requires_objective_gate"
    assert manifest["ralph_ai_delivery_contract"]["status"] == "requires_objective_gate"
    assert manifest["ralph_ai_delivery_contract"]["result_delivery_allowed"] is False
    assert manifest["quality_gate"]["status"] == "required"
    assert manifest["listen"]["status"] == "diagnostic_published"
    assert manifest["listen"]["reason"] == "objective_quality_gate_required"
    assert (listen_dir / "candidate.flac").exists()
    assert (listen_dir / "02_sparse_percussive_source_aux.flac").exists()
