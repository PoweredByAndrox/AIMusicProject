import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "apply_no_reference_pair_reconstruction_vocal_repair.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "apply_no_reference_pair_reconstruction_vocal_repair", MODULE_PATH
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


def test_build_pair_reconstruction_vocal_preserves_raw_mix_sum() -> None:
    module = _load_module()
    sample_rate = 8_000
    times = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
    vocal = 0.18 * np.sin(2.0 * np.pi * 440.0 * times)
    music = 0.45 * np.sin(2.0 * np.pi * 110.0 * times)
    raw = _stereo(vocal + music)
    accepted_minus = _stereo(music)

    repaired_vocal, metrics = module.build_pair_reconstruction_vocal(
        raw_audio=raw,
        accepted_minus=accepted_minus,
        residual_strength=1.0,
        max_peak=0.999,
    )

    reconstruction_error = raw - (repaired_vocal + accepted_minus)

    assert np.max(np.abs(reconstruction_error)) < 1.0e-6
    assert np.max(np.abs(repaired_vocal - _stereo(vocal))) < 1.0e-6
    assert metrics["residual_strength"] == 1.0
    assert metrics["reconstruction_error_rms"] < 1.0e-8
    assert metrics["peak_limited"] is False


def test_manifest_is_vocal_candidate_only_and_requires_post_judge() -> None:
    module = _load_module()

    manifest = module.build_manifest(
        track_id="track-a",
        candidate_id="track-a__vocal_pair_reconstruction_repair",
        raw_audio_path=Path("/app/raw.flac"),
        accepted_minus_path=Path("/app/minus.flac"),
        output_path=Path("/app/out/pair_reconstruction_vocal.flac"),
        candidate_row_jsonl=Path("/app/out/candidate_pair.jsonl"),
        sample_rate=44_100,
        frames=44_100,
        metrics={"residual_strength": 1.0, "reconstruction_error_rms": 0.0},
    )

    assert manifest["render_status"] == "rendered_vocal_segment_switch_candidate"
    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "preview_only_not_for_listen_pack"
    assert manifest["release_status"] == "not_release_ready"
    assert manifest["requires_post_render_vocal_judge"] is True
    assert manifest["candidate_row"]["vocal_path"] == "/app/out/pair_reconstruction_vocal.flac"
    assert manifest["candidate_row"]["instrumental_path"] == "/app/minus.flac"
    assert manifest["candidate_row"]["candidate_family"] == "no_reference_pair_reconstruction_vocal_repair"
    assert manifest["safety"]["training_started"] is False
    assert manifest["safety"]["models_changed"] is False
    assert manifest["safety"]["thresholds_relaxed"] is False


def test_run_build_writes_vocal_candidate_row_and_manifest(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    times = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
    vocal = _stereo(0.15 * np.sin(2.0 * np.pi * 440.0 * times))
    music = _stereo(0.35 * np.sin(2.0 * np.pi * 120.0 * times))
    raw = vocal + music
    raw_path = tmp_path / "raw.flac"
    minus_path = tmp_path / "minus.flac"
    output_dir = tmp_path / "out"
    _write_audio(raw_path, raw, sample_rate)
    _write_audio(minus_path, music, sample_rate)

    manifest = module.run_build(
        Namespace(
            raw_audio_path=raw_path,
            accepted_minus_path=minus_path,
            output_dir=output_dir,
            candidate_id="candidate-a",
            track_id="track-a",
            residual_strength=1.0,
            max_peak=0.999,
        )
    )

    output_path = output_dir / "pair_reconstruction_vocal.flac"
    candidate_row_path = output_dir / "candidate_pair.jsonl"
    manifest_path = output_dir / "pair_reconstruction_vocal_repair_manifest.json"
    row = json.loads(candidate_row_path.read_text(encoding="utf-8").strip())
    rendered, rendered_rate = sf.read(str(output_path), always_2d=True, dtype="float32")

    assert rendered_rate == sample_rate
    assert np.max(np.abs(rendered - vocal)) < 1.0e-5
    assert manifest["candidate_row_jsonl"] == str(candidate_row_path)
    assert manifest_path.exists()
    assert row["candidate_id"] == "candidate-a__vocal_pair_reconstruction_repair"
    assert row["delivery_allowed"] is False
