from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tools.analyze_model_split_reconstruction_residual import localize_candidate_reconstruction_residual
from tools.build_training_first_root_cause_queue import build_root_cause_queue
from tools.stem_region_repair_common import ISSUE_BAD_RECONSTRUCTION, write_audio


SAMPLE_RATE = 8_000


def _sine(freq: float, seconds: float = 0.4, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    mono = amp * np.sin(2.0 * math.pi * freq * t)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _candidate_with_drum_residual(tmp_path: Path, candidate_id: str = "c001") -> dict[str, object]:
    vocals = _sine(220.0, amp=0.08)
    minus = _sine(110.0, amp=0.07)
    drums = np.zeros_like(vocals)
    drums[500:530, :] = 0.18
    drums[1300:1330, :] = -0.16
    mix = vocals + minus + drums
    audio_dir = tmp_path / candidate_id / "stems"
    paths = {
        "vocals": write_audio(audio_dir / "vocals.flac", vocals, SAMPLE_RATE),
        "minus": write_audio(audio_dir / "minus.flac", minus, SAMPLE_RATE),
        "drums": write_audio(audio_dir / "drums.flac", drums, SAMPLE_RATE),
    }
    mix_path = write_audio(tmp_path / candidate_id / "mix.flac", mix, SAMPLE_RATE)
    return {
        "candidate_id": candidate_id,
        "window_id": "w000",
        "verdict": "blocked_bad_reconstruction",
        "mix_path": str(mix_path),
        "stem_paths": {stem: str(path) for stem, path in paths.items()},
        "reconstruction": {"delivery_pair": {"blockers": ["mix_conservation_residual"]}},
        "target_specific_composition": {
            "routes": [
                {"lane": "stem_lane", "route_id": "hybrid_htdemucs_ft_bs_roformer_6s"},
                {"lane": "minus_lane", "route_id": "mdx23c_vocals:instrumental"},
            ]
        },
    }


def test_bad_reconstruction_residual_localizes_to_source_stem(tmp_path: Path) -> None:
    candidate = _candidate_with_drum_residual(tmp_path)

    result = localize_candidate_reconstruction_residual(candidate, output_dir=tmp_path / "loc")

    assert result["schema"] == "model_split_reconstruction_residual.v1"
    assert result["verdict"] == "localized"
    assert result["issue_type"] == ISSUE_BAD_RECONSTRUCTION
    assert result["affected_stem"] == "drums"
    assert result["residual_region"]["start_sec"] < result["residual_region"]["end_sec"]
    assert result["projection_scores"]["drums"] > result["projection_scores"]["vocals"]


def test_root_cause_queue_uses_residual_localization_for_bad_reconstruction(tmp_path: Path) -> None:
    candidate = _candidate_with_drum_residual(tmp_path)
    run_json = tmp_path / "run.json"
    run_json.write_text(json.dumps({"schema": "run.v1", "candidates": [candidate]}), encoding="utf-8")

    manifest = build_root_cause_queue(run_json_path=run_json, output_dir=tmp_path / "queue", min_repeats=1)
    rows = [
        json.loads(line)
        for line in Path(manifest["queue_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    assert rows[0]["issue_type"] == ISSUE_BAD_RECONSTRUCTION
    assert rows[0]["affected_stem"] == "drums"
    assert rows[0]["training_action"] == "mine_reconstruction_repair_example"
    assert rows[0]["bad_stem_path"].endswith("drums.flac")
    assert rows[0]["residual_localization"]["verdict"] == "localized"


def test_infra_gap_rows_quarantine_route_for_next_run(tmp_path: Path) -> None:
    run_json = tmp_path / "run.json"
    run_json.write_text(
        json.dumps(
            {
                "schema": "run.v1",
                "candidates": [
                    {
                        "candidate_id": "c404",
                        "window_id": "w000",
                        "verdict": "blocked_unknown",
                        "execution_status": "failed_runtime_error",
                        "error": {"type": "RuntimeError", "message": "model unavailable"},
                        "target_specific_composition": {
                            "routes": [{"lane": "vocal_lane", "route_id": "broken_vocal_model"}]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_root_cause_queue(run_json_path=run_json, output_dir=tmp_path / "queue", min_repeats=2)
    row = json.loads(Path(manifest["queue_jsonl"]).read_text(encoding="utf-8").splitlines()[0])

    assert row["root_cause"] == "infra_gap"
    assert row["quarantined_for_next_run"] is True
    assert row["training_action"] == "fix_or_disable_infra_route"


def test_infra_gap_extracts_failed_registered_model_route_signature(tmp_path: Path) -> None:
    run_json = tmp_path / "run.json"
    run_json.write_text(
        json.dumps(
            {
                "schema": "run.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "window_id": "w000",
                        "verdict": "blocked_unknown",
                        "execution_status": "failed_runtime_error",
                        "error": {
                            "type": "CalledProcessError",
                            "message": (
                                "Command '['python', 'tools/run_registered_model_candidate.py', "
                                "'/tmp/mix.flac', '/tmp/out', '--model-id', 'bs_mamba2_vocals', "
                                "'--device', 'cuda', '--target-stem', 'vocals']' returned non-zero exit status 1."
                            ),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_root_cause_queue(run_json_path=run_json, output_dir=tmp_path / "queue", min_repeats=2)
    row = json.loads(Path(manifest["queue_jsonl"]).read_text(encoding="utf-8").splitlines()[0])

    assert row["route_signature"] == "vocal_lane:bs_mamba2_vocals"
    assert row["quarantined_for_next_run"] is True
