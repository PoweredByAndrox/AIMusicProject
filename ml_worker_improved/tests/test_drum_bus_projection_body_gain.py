from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools import drum_bus_projection_body_gain as drum_route


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_24")


def _synthetic_stems(sample_rate: int = 16000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    duration_sec = 2.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    kick = np.zeros_like(t)
    kick[:: sample_rate // 4] = 1.0
    kick = np.convolve(kick, np.hanning(96).astype(np.float32), mode="same")
    snare = np.zeros_like(t)
    snare[sample_rate // 8 :: sample_rate // 2] = 0.55
    snare = np.convolve(snare, np.hanning(64).astype(np.float32), mode="same")
    drum_body = 0.34 * kick + 0.18 * snare
    tonal_leak = 0.08 * np.sin(2.0 * np.pi * 660.0 * t).astype(np.float32)
    high_leak = 0.035 * np.sin(2.0 * np.pi * 2400.0 * t).astype(np.float32)
    base = drum_body + tonal_leak + high_leak
    leakage = tonal_leak * 1.8 + high_leak
    body_reference = drum_body * 1.18 + 0.02 * np.sin(2.0 * np.pi * 180.0 * t).astype(np.float32)
    raw_ref = base + 0.1 * np.sin(2.0 * np.pi * 110.0 * t).astype(np.float32)
    return (
        np.stack([base, base], axis=1),
        np.stack([body_reference, body_reference], axis=1),
        np.stack([leakage, leakage], axis=1),
        np.stack([raw_ref, raw_ref], axis=1),
    )


def test_route_writes_manifest_with_final_output_metrics_from_reloaded_flac(tmp_path: Path) -> None:
    sample_rate = 16000
    base, body_reference, leakage, raw_ref = _synthetic_stems(sample_rate)
    base_path = tmp_path / "base.flac"
    body_reference_path = tmp_path / "body_ref.flac"
    leakage_path = tmp_path / "other.flac"
    raw_ref_path = tmp_path / "raw_ref.flac"
    _write_audio(base_path, base, sample_rate)
    _write_audio(body_reference_path, body_reference, sample_rate)
    _write_audio(leakage_path, leakage, sample_rate)
    _write_audio(raw_ref_path, raw_ref, sample_rate)

    manifest = drum_route.build_drum_bus_projection_body_gain(
        base_drums_path=base_path,
        body_reference_drums_path=body_reference_path,
        leakage_stem_paths=[leakage_path],
        output_dir=tmp_path / "out",
        raw_ref_path=raw_ref_path,
        listen_dir=tmp_path / "listen",
        validation_windows=[drum_route.ValidationWindow("first_half", 0.0, 1.0)],
        config=drum_route.DrumRouteConfig(max_body_gain=1.04, max_peak=0.98),
        source_label="01_base.flac",
        debleed_label="02_debleed.flac",
        output_label="03_output.flac",
        body_reference_label="04_body_ref.flac",
        command_args=["base.flac", "body_ref.flac", "out"],
    )

    output_path = Path(manifest["outputs"]["output_path"])
    reloaded_audio, reloaded_sample_rate = sf.read(str(output_path), always_2d=True, dtype="float32")
    saved_manifest = json.loads(Path(manifest["outputs"]["manifest_path"]).read_text(encoding="utf-8"))
    candidate_manifest_path = output_path.parent / "stem_manifest.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))

    assert reloaded_sample_rate == sample_rate
    assert saved_manifest["contract"]["accepted_by_listening"] is False
    assert saved_manifest["contract"]["not_timestamp_hardcoded"] is True
    assert saved_manifest["contract"]["fallback_used"] is False
    assert candidate_manifest["contract"]["benchmark_compatible"] is True
    assert candidate_manifest["contract"]["target_reference_assisted"] is False
    assert candidate_manifest["contract"]["target_reference_used_at_application"] is False
    assert candidate_manifest["contract"]["application_reference_free"] is True
    assert candidate_manifest["contract"]["delivery_allowed_without_reference"] is True
    assert candidate_manifest["contract"]["fallback_used"] is False
    assert candidate_manifest["contract"]["per_track_hardcode_allowed"] is False
    assert candidate_manifest["contract"]["not_timestamp_hardcoded"] is True
    assert candidate_manifest["stems"]["drums"]["path"] == str(output_path)
    assert saved_manifest["selection"]["selected"] in {"debleed_intermediate", "dynamic_body_gain"}
    assert "candidate_headroom_gain" in saved_manifest["selection"]
    assert saved_manifest["metrics"]["final_output"]["sha256"] == drum_route._sha256(output_path)
    assert saved_manifest["metrics"]["final_output"]["frames"] == reloaded_audio.shape[0]
    assert saved_manifest["listen_pack"]["candidate_map"]["03_output.flac"].endswith("03_output.flac")


def test_validation_windows_do_not_change_audio_output(tmp_path: Path) -> None:
    sample_rate = 16000
    base, body_reference, leakage, _raw_ref = _synthetic_stems(sample_rate)
    base_path = tmp_path / "base.flac"
    body_reference_path = tmp_path / "body_ref.flac"
    leakage_path = tmp_path / "other.flac"
    _write_audio(base_path, base, sample_rate)
    _write_audio(body_reference_path, body_reference, sample_rate)
    _write_audio(leakage_path, leakage, sample_rate)

    common_kwargs = {
        "base_drums_path": base_path,
        "body_reference_drums_path": body_reference_path,
        "leakage_stem_paths": [leakage_path],
        "raw_ref_path": None,
        "listen_dir": None,
        "config": drum_route.DrumRouteConfig(max_body_gain=1.04, max_peak=0.98),
        "source_label": "01_base.flac",
        "debleed_label": "02_debleed.flac",
        "output_label": "03_output.flac",
        "body_reference_label": "04_body_ref.flac",
        "command_args": [],
    }
    first = drum_route.build_drum_bus_projection_body_gain(
        output_dir=tmp_path / "out_a",
        validation_windows=[drum_route.ValidationWindow("a", 0.0, 0.75)],
        **common_kwargs,
    )
    second = drum_route.build_drum_bus_projection_body_gain(
        output_dir=tmp_path / "out_b",
        validation_windows=[drum_route.ValidationWindow("b", 1.0, 1.75)],
        **common_kwargs,
    )

    assert first["metrics"]["final_output"]["sha256"] == second["metrics"]["final_output"]["sha256"]


def test_route_reduces_synthetic_leakage_correlation(tmp_path: Path) -> None:
    sample_rate = 16000
    base, body_reference, leakage, _raw_ref = _synthetic_stems(sample_rate)
    base_path = tmp_path / "base.flac"
    body_reference_path = tmp_path / "body_ref.flac"
    leakage_path = tmp_path / "other.flac"
    _write_audio(base_path, base, sample_rate)
    _write_audio(body_reference_path, body_reference, sample_rate)
    _write_audio(leakage_path, leakage, sample_rate)

    manifest = drum_route.build_drum_bus_projection_body_gain(
        base_drums_path=base_path,
        body_reference_drums_path=body_reference_path,
        leakage_stem_paths=[leakage_path],
        output_dir=tmp_path / "out",
        raw_ref_path=None,
        listen_dir=None,
        validation_windows=[],
        config=drum_route.DrumRouteConfig(max_body_gain=1.04, max_peak=0.98),
        source_label="01_base.flac",
        debleed_label="02_debleed.flac",
        output_label="03_output.flac",
        body_reference_label="04_body_ref.flac",
    )

    assert (
        manifest["metrics"]["debleed_intermediate"]["max_ref_corr_to_non_drum_buses"]
        < manifest["metrics"]["base_drums"]["max_ref_corr_to_non_drum_buses"]
    )


def test_body_gain_rejection_selects_debleed_when_headroom_would_reduce_output(tmp_path: Path) -> None:
    sample_rate = 16000
    base, body_reference, leakage, _raw_ref = _synthetic_stems(sample_rate)
    base_path = tmp_path / "base.flac"
    body_reference_path = tmp_path / "body_ref.flac"
    leakage_path = tmp_path / "other.flac"
    _write_audio(base_path, base * 0.9, sample_rate)
    _write_audio(body_reference_path, body_reference * 6.0, sample_rate)
    _write_audio(leakage_path, leakage * 0.3, sample_rate)

    manifest = drum_route.build_drum_bus_projection_body_gain(
        base_drums_path=base_path,
        body_reference_drums_path=body_reference_path,
        leakage_stem_paths=[leakage_path],
        output_dir=tmp_path / "out",
        raw_ref_path=None,
        listen_dir=tmp_path / "listen",
        validation_windows=[],
        config=drum_route.DrumRouteConfig(max_body_gain=2.0, max_peak=0.2),
        source_label="01_base.flac",
        debleed_label="02_debleed.flac",
        output_label="03_selected.flac",
        body_reference_label="04_body_ref.flac",
    )

    output_path = Path(manifest["outputs"]["output_path"])
    debleed_path = Path(manifest["outputs"]["debleed_intermediate_path"])
    readme = (tmp_path / "listen" / "00_README.txt").read_text(encoding="utf-8")
    assert manifest["selection"]["selected"] == "debleed_intermediate"
    assert manifest["selection"]["reason"] == "dynamic_body_gain_requires_global_headroom_reduction"
    assert output_path.name == "drums_bus_projection_selected_debleed.flac"
    assert drum_route._sha256(output_path) == drum_route._sha256(debleed_path)
    assert "final selected output (debleed_intermediate)" in readme
    assert "dynamic_body_gain_requires_global_headroom_reduction" in readme


def test_main_writes_cli_provenance_and_listen_labels(tmp_path: Path) -> None:
    sample_rate = 16000
    base, body_reference, leakage, raw_ref = _synthetic_stems(sample_rate)
    base_path = tmp_path / "base.flac"
    body_reference_path = tmp_path / "body_ref.flac"
    leakage_path = tmp_path / "other.flac"
    raw_ref_path = tmp_path / "raw_ref.flac"
    output_dir = tmp_path / "out"
    listen_dir = tmp_path / "listen"
    _write_audio(base_path, base, sample_rate)
    _write_audio(body_reference_path, body_reference, sample_rate)
    _write_audio(leakage_path, leakage, sample_rate)
    _write_audio(raw_ref_path, raw_ref, sample_rate)

    argv = [
        "drum_bus_projection_body_gain.py",
        str(base_path),
        str(body_reference_path),
        str(output_dir),
        "--leakage-stem-path",
        str(leakage_path),
        "--raw-ref-path",
        str(raw_ref_path),
        "--validation-window",
        "target:0:1",
        "--listen-dir",
        str(listen_dir),
        "--source-label",
        "01_base_custom.flac",
        "--debleed-label",
        "02_debleed_custom.flac",
        "--output-label",
        "03_output_custom.flac",
        "--body-reference-label",
        "04_body_ref_custom.flac",
    ]
    with patch.object(sys, "argv", argv):
        assert drum_route.main() == 0

    manifest = json.loads((output_dir / "drum_bus_projection_body_gain_manifest.json").read_text(encoding="utf-8"))
    candidate_map = json.loads((listen_dir / "candidate_map.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["command_args"][0] == str(base_path)
    assert "target" in manifest["metrics"]["validation_windows"]
    assert sorted(candidate_map) == [
        "00_raw_ref.flac",
        "01_base_custom.flac",
        "02_debleed_custom.flac",
        "03_output_custom.flac",
        "04_body_ref_custom.flac",
    ]


def test_route_fails_fast_without_leakage_stems(tmp_path: Path) -> None:
    sample_rate = 16000
    base, body_reference, _leakage, _raw_ref = _synthetic_stems(sample_rate)
    base_path = tmp_path / "base.flac"
    body_reference_path = tmp_path / "body_ref.flac"
    _write_audio(base_path, base, sample_rate)
    _write_audio(body_reference_path, body_reference, sample_rate)

    with pytest.raises(ValueError, match="At least one --leakage-stem-path"):
        drum_route.build_drum_bus_projection_body_gain(
            base_drums_path=base_path,
            body_reference_drums_path=body_reference_path,
            leakage_stem_paths=[],
            output_dir=tmp_path / "out",
            raw_ref_path=None,
            listen_dir=None,
            validation_windows=[],
            config=drum_route.DrumRouteConfig(),
            source_label="01_base.flac",
            debleed_label="02_debleed.flac",
            output_label="03_output.flac",
            body_reference_label="04_body_ref.flac",
        )


def test_parse_validation_window_rejects_invalid_bounds() -> None:
    with pytest.raises(Exception):
        drum_route.parse_validation_window("bad:5:1")


def test_route_fails_fast_on_sample_rate_mismatch(tmp_path: Path) -> None:
    base, body_reference, leakage, _raw_ref = _synthetic_stems(16000)
    base_path = tmp_path / "base.flac"
    body_reference_path = tmp_path / "body_ref.flac"
    leakage_path = tmp_path / "other.flac"
    _write_audio(base_path, base, 16000)
    _write_audio(body_reference_path, body_reference, 22050)
    _write_audio(leakage_path, leakage, 16000)

    with pytest.raises(ValueError, match="Sample rate mismatch"):
        drum_route.build_drum_bus_projection_body_gain(
            base_drums_path=base_path,
            body_reference_drums_path=body_reference_path,
            leakage_stem_paths=[leakage_path],
            output_dir=tmp_path / "out",
            raw_ref_path=None,
            listen_dir=None,
            validation_windows=[],
            config=drum_route.DrumRouteConfig(),
            source_label="01_base.flac",
            debleed_label="02_debleed.flac",
            output_label="03_output.flac",
            body_reference_label="04_body_ref.flac",
        )
