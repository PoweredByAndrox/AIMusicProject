from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "reference_guided_drum_candidate_router.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("reference_guided_drum_candidate_router", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stereo(audio: np.ndarray) -> np.ndarray:
    return np.stack([audio, audio], axis=1).astype(np.float32)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_24")


def _sdr(reference: np.ndarray, candidate: np.ndarray) -> float:
    n = min(reference.shape[0], candidate.shape[0])
    reference = reference[:n]
    candidate = candidate[:n]
    error = candidate - reference
    return float(10.0 * np.log10((np.sum(reference * reference) + 1.0e-12) / (np.sum(error * error) + 1.0e-12)))


def test_router_selects_best_candidate_by_segment_and_marks_reference_assisted(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate * 2, dtype=np.float64) / float(sample_rate)
    first = 0.4 * np.sin(2.0 * np.pi * 120.0 * time[:sample_rate])
    second = 0.4 * np.sin(2.0 * np.pi * 220.0 * time[:sample_rate])
    reference = _stereo(np.concatenate([first, second]))
    candidate_a = reference.copy()
    candidate_b = reference.copy()
    candidate_a[sample_rate:] *= 0.1
    candidate_b[:sample_rate] *= 0.1
    reference_path = tmp_path / "drums_ref.flac"
    candidate_a_path = tmp_path / "candidate_a.flac"
    candidate_b_path = tmp_path / "candidate_b.flac"
    _write_audio(reference_path, reference, sample_rate)
    _write_audio(candidate_a_path, candidate_a, sample_rate)
    _write_audio(candidate_b_path, candidate_b, sample_rate)

    manifest = module.build_reference_guided_drum_candidate_route(
        target_reference_path=reference_path,
        candidate_paths={"candidate_a": candidate_a_path, "candidate_b": candidate_b_path},
        output_dir=tmp_path / "out",
        listen_dir=tmp_path / "listen",
        segment_seconds=1.0,
        crossfade_seconds=0.0,
        max_peak=0.999,
        command_args=[],
    )

    output_audio, _ = module.read_audio(Path(manifest["stems"]["drums"]["path"]))
    assert _sdr(reference, output_audio) > _sdr(reference, candidate_a) + 6.0
    assert manifest["diagnostics"]["selected_counts"] == {"candidate_a": 1, "candidate_b": 1}
    assert manifest["contract"]["target_reference_assisted"] is True
    assert manifest["contract"]["target_reference_used_at_application"] is True
    assert manifest["contract"]["application_reference_free"] is False
    assert manifest["contract"]["no_reference_production_claim"] is False
    assert manifest["route"]["copies_target_reference"] is False
    assert (tmp_path / "listen" / "01_reference_guided_segment_router_drums.flac").exists()


def test_router_rejects_empty_candidate_set(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    reference_path = tmp_path / "drums_ref.flac"
    _write_audio(reference_path, _stereo(np.zeros(sample_rate, dtype=np.float32)), sample_rate)

    with pytest.raises(ValueError, match="at least one candidate"):
        module.build_reference_guided_drum_candidate_route(
            target_reference_path=reference_path,
            candidate_paths={},
            output_dir=tmp_path / "out",
            listen_dir=None,
            segment_seconds=1.0,
            crossfade_seconds=0.0,
            max_peak=0.999,
            command_args=[],
        )


def test_router_crossfade_does_not_insert_silence_at_switch(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 1_000
    reference = _stereo(np.ones(sample_rate * 2, dtype=np.float32) * 0.5)
    candidate_a = reference.copy()
    candidate_b = reference.copy()
    candidate_a[sample_rate:] *= 0.1
    candidate_b[:sample_rate] *= 0.1
    output, _decisions, _counts = module.route_candidate_segments(
        target_reference=reference,
        candidates={"candidate_a": candidate_a, "candidate_b": candidate_b},
        sample_rate=sample_rate,
        segment_seconds=1.0,
        crossfade_seconds=0.05,
    )

    boundary = output[sample_rate - 10:sample_rate + 10]
    assert float(np.min(np.abs(boundary))) > 0.2


def test_equal_power_fade_single_frame_does_not_double_amplitude() -> None:
    module = _load_module()

    fade_out, fade_in = module.equal_power_fade(1)

    assert fade_out.tolist() == [0.0]
    assert fade_in.tolist() == [1.0]


def test_router_cli_smoke(tmp_path: Path) -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    reference = _stereo(0.4 * np.sin(2.0 * np.pi * 120.0 * time))
    candidate = reference * 0.8
    reference_path = tmp_path / "drums_ref.flac"
    candidate_path = tmp_path / "candidate.flac"
    output_dir = tmp_path / "out"
    _write_audio(reference_path, reference, sample_rate)
    _write_audio(candidate_path, candidate, sample_rate)

    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--target-reference",
            str(reference_path),
            "--candidate",
            f"candidate={candidate_path}",
            "--output-dir",
            str(output_dir),
            "--segment-seconds",
            "1.0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    manifest = json.loads(Path(payload["manifest_path"]).read_text(encoding="utf-8"))
    assert Path(manifest["stems"]["drums"]["path"]).exists()
    assert manifest["contract"]["target_reference_assisted"] is True
