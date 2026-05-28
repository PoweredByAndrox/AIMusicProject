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
MODULE_PATH = TOOLS_ROOT / "reference_guided_drum_reconstructor.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("reference_guided_drum_reconstructor", MODULE_PATH)
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


def test_projection_gain_solver_recovers_target_gain() -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    drums = _stereo(0.4 * np.sin(2.0 * np.pi * 110.0 * time))
    melody = _stereo(0.2 * np.sin(2.0 * np.pi * 440.0 * time))
    source_mix = 0.75 * drums + 0.30 * melody

    gains = module.solve_projection_gains(
        source_mix_audio=source_mix,
        target_reference_audio=drums,
        non_target_reference_stems={"melody": melody},
        stem_name="drums",
        ridge=0.0,
    )

    assert gains["drums"] == pytest.approx(0.75, abs=1.0e-4)
    assert gains["melody"] == pytest.approx(0.30, abs=1.0e-4)


def test_reference_locked_teacher_outputs_exact_target(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    drums = _stereo(0.4 * np.sin(2.0 * np.pi * 110.0 * time))
    melody = _stereo(0.2 * np.sin(2.0 * np.pi * 440.0 * time))
    source_mix = 0.75 * drums + 0.30 * melody
    source_path = tmp_path / "mix.flac"
    drums_path = tmp_path / "drums.flac"
    melody_path = tmp_path / "melody.flac"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(drums_path, drums, sample_rate)
    _write_audio(melody_path, melody, sample_rate)

    manifest = module.build_reference_guided_drum_reconstruction(
        source_mix_path=source_path,
        target_reference_path=drums_path,
        non_target_reference_paths={"melody": melody_path},
        output_dir=tmp_path / "out",
        listen_dir=None,
        stem_name="drums",
        reference_lock_strength=1.0,
        ridge=0.0,
        command_args=["--reference-lock-strength", "1.0"],
    )

    output_audio, _ = module.read_audio(Path(manifest["stems"]["drums"]["path"]))
    expected_audio, _ = module.read_audio(drums_path)
    assert np.max(np.abs(output_audio - expected_audio)) <= 1.0e-4
    assert manifest["stems"]["drums"]["sha256"] == module.sha256_file(drums_path)
    assert manifest["diagnostics"]["exact_reference_copy"] is True
    assert manifest["contract"]["target_reference_assisted"] is True
    assert manifest["contract"]["no_reference_production_claim"] is False
    assert manifest["projection_gains"]["drums"] == pytest.approx(0.75, abs=1.0e-4)


def test_reconstructor_rejects_target_reference_as_non_target(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    audio = _stereo(np.zeros(sample_rate, dtype=np.float32))
    source_path = tmp_path / "mix.flac"
    drums_path = tmp_path / "drums.flac"
    _write_audio(source_path, audio, sample_rate)
    _write_audio(drums_path, audio, sample_rate)

    with pytest.raises(ValueError, match="must not include target stem"):
        module.build_reference_guided_drum_reconstruction(
            source_mix_path=source_path,
            target_reference_path=drums_path,
            non_target_reference_paths={"drums": drums_path},
            output_dir=tmp_path / "out",
            listen_dir=None,
            stem_name="drums",
            reference_lock_strength=1.0,
            ridge=1.0e-8,
            command_args=[],
        )


def test_reference_locked_teacher_metadata_matches_exact_copy_with_uneven_lengths(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    source_audio = _stereo(np.zeros(sample_rate, dtype=np.float32))
    drums_audio = _stereo(np.zeros(sample_rate + 128, dtype=np.float32))
    melody_audio = _stereo(np.zeros(sample_rate, dtype=np.float32))
    source_path = tmp_path / "mix.flac"
    drums_path = tmp_path / "drums.flac"
    melody_path = tmp_path / "melody.flac"
    _write_audio(source_path, source_audio, sample_rate)
    _write_audio(drums_path, drums_audio, sample_rate)
    _write_audio(melody_path, melody_audio, sample_rate)

    manifest = module.build_reference_guided_drum_reconstruction(
        source_mix_path=source_path,
        target_reference_path=drums_path,
        non_target_reference_paths={"melody": melody_path},
        output_dir=tmp_path / "out",
        listen_dir=None,
        stem_name="drums",
        reference_lock_strength=1.0,
        ridge=1.0e-8,
        command_args=[],
    )

    assert manifest["diagnostics"]["frames"] == manifest["stems"]["drums"]["frames"]
    assert manifest["diagnostics"]["analysis_frames"] == sample_rate
    assert manifest["stems"]["drums"]["frames"] == sample_rate + 128


def test_projection_gain_solver_falls_back_for_singular_matrix() -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    drums = _stereo(0.4 * np.sin(2.0 * np.pi * 110.0 * time))
    duplicated = drums.copy()
    source_mix = 0.75 * drums + 0.30 * duplicated

    gains = module.solve_projection_gains(
        source_mix_audio=source_mix,
        target_reference_audio=drums,
        non_target_reference_stems={"duplicated": duplicated},
        stem_name="drums",
        ridge=0.0,
    )

    assert gains["drums"] + gains["duplicated"] == pytest.approx(1.05, abs=1.0e-4)


def test_projection_gain_solution_reports_singular_matrix_fallback() -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    drums = _stereo(0.4 * np.sin(2.0 * np.pi * 110.0 * time))
    duplicated = drums.copy()
    source_mix = 0.75 * drums + 0.30 * duplicated

    solution = module.solve_projection_gain_solution(
        source_mix_audio=source_mix,
        target_reference_audio=drums,
        non_target_reference_stems={"duplicated": duplicated},
        stem_name="drums",
        ridge=0.0,
    )

    assert solution.fallback_used is True
    assert solution.solve_method == "least_squares_after_singular_matrix"
    assert solution.gains["drums"] + solution.gains["duplicated"] == pytest.approx(1.05, abs=1.0e-4)


def test_reconstructor_manifest_reports_singular_matrix_fallback(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    drums = _stereo(0.4 * np.sin(2.0 * np.pi * 110.0 * time))
    duplicated = drums.copy()
    source_mix = 0.75 * drums + 0.30 * duplicated
    source_path = tmp_path / "mix.flac"
    drums_path = tmp_path / "drums.flac"
    duplicated_path = tmp_path / "duplicated.flac"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(drums_path, drums, sample_rate)
    _write_audio(duplicated_path, duplicated, sample_rate)

    manifest = module.build_reference_guided_drum_reconstruction(
        source_mix_path=source_path,
        target_reference_path=drums_path,
        non_target_reference_paths={"duplicated": duplicated_path},
        output_dir=tmp_path / "out",
        listen_dir=None,
        stem_name="drums",
        reference_lock_strength=1.0,
        ridge=0.0,
        command_args=[],
    )

    assert manifest["contract"]["fallback_used"] is True
    assert manifest["diagnostics"]["projection_solve_method"] == "least_squares_after_singular_matrix"


def test_non_exact_teacher_preserves_full_target_reference_length(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    source_audio = _stereo(np.zeros(sample_rate, dtype=np.float32))
    drums_audio = _stereo(np.zeros(sample_rate + 128, dtype=np.float32))
    melody_audio = _stereo(np.zeros(sample_rate, dtype=np.float32))
    source_path = tmp_path / "mix.flac"
    drums_path = tmp_path / "drums.flac"
    melody_path = tmp_path / "melody.flac"
    _write_audio(source_path, source_audio, sample_rate)
    _write_audio(drums_path, drums_audio, sample_rate)
    _write_audio(melody_path, melody_audio, sample_rate)

    manifest = module.build_reference_guided_drum_reconstruction(
        source_mix_path=source_path,
        target_reference_path=drums_path,
        non_target_reference_paths={"melody": melody_path},
        output_dir=tmp_path / "out",
        listen_dir=None,
        stem_name="drums",
        reference_lock_strength=0.5,
        ridge=1.0e-8,
        command_args=[],
    )

    assert manifest["diagnostics"]["analysis_frames"] == sample_rate
    assert manifest["stems"]["drums"]["frames"] == sample_rate + 128
    assert manifest["diagnostics"]["frames"] == sample_rate + 128


def test_reconstructor_cli_smoke(tmp_path: Path) -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    drums = _stereo(0.4 * np.sin(2.0 * np.pi * 110.0 * time))
    melody = _stereo(0.2 * np.sin(2.0 * np.pi * 440.0 * time))
    source_mix = 0.75 * drums + 0.30 * melody
    source_path = tmp_path / "mix.flac"
    drums_path = tmp_path / "drums.flac"
    melody_path = tmp_path / "melody.flac"
    output_dir = tmp_path / "out"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(drums_path, drums, sample_rate)
    _write_audio(melody_path, melody, sample_rate)

    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--source-mix",
            str(source_path),
            "--target-reference",
            str(drums_path),
            "--non-target-reference",
            f"melody={melody_path}",
            "--output-dir",
            str(output_dir),
            "--stem-name",
            "drums",
            "--reference-lock-strength",
            "1.0",
            "--ridge",
            "0.0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
    assert Path(payload["output"]["path"]).exists()
    assert manifest["projection_gains"]["drums"] == pytest.approx(0.75, abs=1.0e-4)
    assert manifest["provenance"]["command_args"]
