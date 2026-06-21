import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "apply_no_reference_pair_reconstruction_repair.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "apply_no_reference_pair_reconstruction_repair", MODULE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stereo(audio: np.ndarray) -> np.ndarray:
    return np.column_stack([audio, audio]).astype(np.float32)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), audio, sample_rate, subtype="PCM_24")


def test_build_pair_reconstruction_minus_preserves_raw_mix_sum() -> None:
    module = _load_module()
    sample_rate = 8_000
    times = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
    vocal = 0.18 * np.sin(2.0 * np.pi * 440.0 * times)
    music = 0.45 * np.sin(2.0 * np.pi * 110.0 * times)
    raw = _stereo(vocal + music)
    selected_vocal = _stereo(vocal)

    minus, metrics = module.build_pair_reconstruction_minus(
        raw_audio=raw,
        selected_vocal=selected_vocal,
        residual_strength=1.0,
        max_peak=0.999,
    )

    reconstruction_error = raw - (minus + selected_vocal)

    assert np.max(np.abs(reconstruction_error)) < 1.0e-6
    assert np.max(np.abs(minus - _stereo(music))) < 1.0e-6
    assert metrics["residual_strength"] == 1.0
    assert metrics["reconstruction_error_rms"] < 1.0e-8
    assert metrics["peak_limited"] is False


def test_build_pair_reconstruction_minus_can_be_strength_limited() -> None:
    module = _load_module()
    sample_rate = 8_000
    times = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
    vocal = _stereo(0.20 * np.sin(2.0 * np.pi * 440.0 * times))
    music = _stereo(0.30 * np.sin(2.0 * np.pi * 120.0 * times))
    raw = vocal + music

    minus, metrics = module.build_pair_reconstruction_minus(
        raw_audio=raw,
        selected_vocal=vocal,
        residual_strength=0.5,
        max_peak=0.999,
    )

    expected = 0.5 * (raw - vocal)

    assert np.max(np.abs(minus - expected)) < 1.0e-6
    assert metrics["residual_strength"] == 0.5
    assert metrics["reconstruction_error_rms"] > 0.0


def test_manifest_is_candidate_only_and_requires_post_judge() -> None:
    module = _load_module()

    manifest = module.build_manifest(
        track_id="track-a",
        candidate_id="track-a__pair_reconstruction_repair",
        raw_audio_path=Path("/app/raw.flac"),
        selected_vocal_path=Path("/app/vocal.flac"),
        output_path=Path("/app/out/pair_reconstruction_minus.flac"),
        candidate_row_jsonl=Path("/app/out/candidate_pair.jsonl"),
        sample_rate=44_100,
        frames=44_100,
        metrics={"residual_strength": 1.0, "reconstruction_error_rms": 0.0},
        base_minus_path=Path("/app/base.flac"),
        windowed_pair_judge_jsonl=Path("/app/pair_judge.jsonl"),
    )

    assert manifest["delivery_status"] == "candidate_only"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "blocked_until_judged"
    assert manifest["release_status"] == "not_release_ready"
    assert manifest["requires_post_judge"] is True
    assert manifest["requires_post_render_minus_self_judge"] is True
    assert manifest["candidate_row"]["instrumental_path"] == "/app/out/pair_reconstruction_minus.flac"
    assert manifest["candidate_row"]["vocal_path"] == "/app/vocal.flac"
    assert manifest["candidate_row"]["base_minus_path"] == "/app/base.flac"
    assert manifest["candidate_row"]["candidate_family"] == "no_reference_pair_reconstruction_repair"
    assert manifest["safety"]["training_started"] is False
    assert manifest["safety"]["models_changed"] is False
    assert manifest["safety"]["thresholds_relaxed"] is False
    assert "do_not_mark_release_ready_from_repair_manifest" in manifest["forbidden_actions"]


def test_run_build_writes_candidate_row_and_manifest(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    times = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
    vocal = _stereo(0.15 * np.sin(2.0 * np.pi * 440.0 * times))
    music = _stereo(0.35 * np.sin(2.0 * np.pi * 120.0 * times))
    raw = vocal + music
    raw_path = tmp_path / "raw.flac"
    vocal_path = tmp_path / "vocal.flac"
    output_dir = tmp_path / "out"
    _write_audio(raw_path, raw, sample_rate)
    _write_audio(vocal_path, vocal, sample_rate)

    manifest = module.run_build(
        Namespace(
            raw_audio_path=raw_path,
            selected_vocal_path=vocal_path,
            output_dir=output_dir,
            base_minus_path=None,
            windowed_pair_judge_jsonl=None,
            candidate_id="candidate-a",
            track_id="track-a",
            residual_strength=1.0,
            max_peak=0.999,
        )
    )

    output_path = output_dir / "pair_reconstruction_minus.flac"
    candidate_row_path = output_dir / "candidate_pair.jsonl"
    manifest_path = output_dir / "pair_reconstruction_repair_manifest.json"
    row = json.loads(candidate_row_path.read_text(encoding="utf-8").strip())
    rendered, rendered_rate = sf.read(str(output_path), always_2d=True, dtype="float32")

    assert rendered_rate == sample_rate
    assert np.max(np.abs(rendered - music)) < 1.0e-5
    assert manifest["candidate_row_jsonl"] == str(candidate_row_path)
    assert manifest_path.exists()
    assert row["candidate_id"] == "candidate-a__pair_reconstruction_repair"
    assert row["delivery_allowed"] is False


def test_run_build_can_write_short_output_names(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    times = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
    vocal = _stereo(0.15 * np.sin(2.0 * np.pi * 440.0 * times))
    music = _stereo(0.35 * np.sin(2.0 * np.pi * 120.0 * times))
    raw = vocal + music
    raw_path = tmp_path / "raw.flac"
    vocal_path = tmp_path / "vocal.flac"
    output_dir = tmp_path / "out"
    _write_audio(raw_path, raw, sample_rate)
    _write_audio(vocal_path, vocal, sample_rate)

    manifest = module.run_build(
        Namespace(
            raw_audio_path=raw_path,
            selected_vocal_path=vocal_path,
            output_dir=output_dir,
            base_minus_path=None,
            windowed_pair_judge_jsonl=None,
            candidate_id="candidate-a",
            track_id="track-a",
            residual_strength=1.0,
            max_peak=0.999,
            short_output_names=True,
        )
    )

    output_path = output_dir / "m001.flac"
    candidate_row_path = output_dir / "rows.jsonl"
    manifest_path = output_dir / "manifest.json"
    row = json.loads(candidate_row_path.read_text(encoding="utf-8").strip())

    assert output_path.exists()
    assert manifest_path.exists()
    assert not (output_dir / "pair_reconstruction_repair_manifest.json").exists()
    assert not (output_dir / "candidate_pair.jsonl").exists()
    assert manifest["manifest_path"] == str(manifest_path)
    assert manifest["candidate_row_jsonl"] == str(candidate_row_path)
    assert manifest["artifacts"]["pair_reconstruction_minus_flac"] == str(output_path)
    assert manifest["short_output_names"] is True
    assert manifest["artifact_name_policy"]["path_segment_limit"] == 32
    assert row["instrumental_path"] == str(output_path)
