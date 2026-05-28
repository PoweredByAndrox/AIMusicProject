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
MODULE_PATH = TOOLS_ROOT / "teacher_dictionary_drum_separator.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("teacher_dictionary_drum_separator", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stereo(audio: np.ndarray) -> np.ndarray:
    return np.stack([audio, audio * 0.92], axis=1).astype(np.float32)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_24")


def _sdr(reference: np.ndarray, candidate: np.ndarray) -> float:
    n = min(reference.shape[0], candidate.shape[0])
    reference = reference[:n]
    candidate = candidate[:n]
    error = candidate - reference
    return float(10.0 * np.log10((np.sum(reference * reference) + 1.0e-12) / (np.sum(error * error) + 1.0e-12)))


def test_dictionary_route_application_is_reference_free_and_improves_over_mix(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate * 2, dtype=np.float64) / float(sample_rate)
    drums = _stereo(0.42 * np.sin(2.0 * np.pi * 120.0 * time) * (0.5 + 0.5 * np.sin(2.0 * np.pi * 8.0 * time) ** 2))
    melody = _stereo(0.35 * np.sin(2.0 * np.pi * 880.0 * time))
    bass = _stereo(0.22 * np.sin(2.0 * np.pi * 55.0 * time))
    source_mix = drums + melody + bass
    source_path = tmp_path / "mix.flac"
    drums_path = tmp_path / "drums.flac"
    melody_path = tmp_path / "melody.flac"
    bass_path = tmp_path / "bass.flac"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(drums_path, drums, sample_rate)
    _write_audio(melody_path, melody, sample_rate)
    _write_audio(bass_path, bass, sample_rate)

    profile_bank_path = tmp_path / "profile_bank.json"
    profile_bank = module.build_teacher_dictionary_profile(
        target_stem_name="drums",
        target_reference_path=drums_path,
        non_target_reference_paths={"melody": melody_path, "bass": bass_path},
        output_profile_bank_path=profile_bank_path,
        config=module.DictionaryBuildConfig(n_fft=512, hop_length=128, max_templates_per_stem=6, active_percentile=60.0),
        command_args=[],
    )
    manifest = module.apply_teacher_dictionary_profile(
        profile_bank_path=profile_bank_path,
        source_mix_path=source_path,
        output_dir=tmp_path / "out",
        listen_dir=tmp_path / "listen",
        apply_config=module.DictionaryApplyConfig(mask_ceiling=0.98, temporal_smoothing_frames=1, allow_missing_holdout=True),
        command_args=[],
    )

    output_audio, _ = module.read_audio(Path(manifest["stems"]["drums"]["path"]))
    mix_sdr = _sdr(drums, source_mix)
    output_sdr = _sdr(drums, output_audio)
    assert output_sdr > mix_sdr + 3.0
    assert profile_bank["contract"]["profile_source_reference_assisted"] is True
    assert profile_bank["contract"]["stores_timestamp_rules"] is False
    assert manifest["contract"]["target_reference_assisted"] is False
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert manifest["contract"]["stores_timestamp_rules"] is False
    assert (tmp_path / "listen" / "01_teacher_dictionary_drums.flac").exists()


def test_dictionary_route_blocks_application_without_holdout_override(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    drums = _stereo(0.4 * np.sin(2.0 * np.pi * 120.0 * time))
    melody = _stereo(0.3 * np.sin(2.0 * np.pi * 880.0 * time))
    source_path = tmp_path / "mix.flac"
    drums_path = tmp_path / "drums.flac"
    melody_path = tmp_path / "melody.flac"
    profile_bank_path = tmp_path / "profile_bank.json"
    _write_audio(source_path, drums + melody, sample_rate)
    _write_audio(drums_path, drums, sample_rate)
    _write_audio(melody_path, melody, sample_rate)
    module.build_teacher_dictionary_profile(
        target_stem_name="drums",
        target_reference_path=drums_path,
        non_target_reference_paths={"melody": melody_path},
        output_profile_bank_path=profile_bank_path,
        config=module.DictionaryBuildConfig(n_fft=512, hop_length=128, max_templates_per_stem=4),
        command_args=[],
    )

    with pytest.raises(ValueError, match="requires passed holdout validation"):
        module.apply_teacher_dictionary_profile(
            profile_bank_path=profile_bank_path,
            source_mix_path=source_path,
            output_dir=tmp_path / "out",
            listen_dir=None,
            apply_config=module.DictionaryApplyConfig(),
            command_args=[],
        )


def test_dictionary_route_rejects_target_in_non_target_paths(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    audio = _stereo(np.zeros(sample_rate, dtype=np.float32))
    drums_path = tmp_path / "drums.flac"
    _write_audio(drums_path, audio, sample_rate)

    with pytest.raises(ValueError, match="must not include target stem"):
        module.build_teacher_dictionary_profile(
            target_stem_name="drums",
            target_reference_path=drums_path,
            non_target_reference_paths={"drums": drums_path},
            output_profile_bank_path=tmp_path / "profile_bank.json",
            config=module.DictionaryBuildConfig(n_fft=512, hop_length=128),
            command_args=[],
        )


def test_dictionary_route_cli_smoke(tmp_path: Path) -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    drums = _stereo(0.4 * np.sin(2.0 * np.pi * 120.0 * time))
    melody = _stereo(0.3 * np.sin(2.0 * np.pi * 880.0 * time))
    source_mix = drums + melody
    source_path = tmp_path / "mix.flac"
    drums_path = tmp_path / "drums.flac"
    melody_path = tmp_path / "melody.flac"
    profile_bank_path = tmp_path / "profile_bank.json"
    output_dir = tmp_path / "out"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(drums_path, drums, sample_rate)
    _write_audio(melody_path, melody, sample_rate)

    build_completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "build-bank",
            "--target-stem-name",
            "drums",
            "--target-reference",
            str(drums_path),
            "--non-target-reference",
            f"melody={melody_path}",
            "--output-profile-bank",
            str(profile_bank_path),
            "--n-fft",
            "512",
            "--hop-length",
            "128",
            "--max-templates-per-stem",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(build_completed.stdout)["contract"]["profile_source_reference_assisted"] is True

    apply_completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "apply-bank",
            "--profile-bank",
            str(profile_bank_path),
            "--source-mix",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--allow-missing-holdout",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(apply_completed.stdout)
    manifest = json.loads(Path(payload["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["contract"]["application_reference_free"] is True
    assert Path(manifest["stems"]["drums"]["path"]).exists()
