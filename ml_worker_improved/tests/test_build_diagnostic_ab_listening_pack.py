from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_diagnostic_ab_listening_pack.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_diagnostic_ab_listening_pack", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_flac(path: Path, values: np.ndarray, sample_rate: int = 8000, subtype: str = "PCM_24") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), values.astype(np.float64), sample_rate, subtype=subtype)


def _write_comparison_manifest(root: Path) -> Path:
    segments = [
        {
            "segment_id": "segment_0001",
            "window_sec": [0.0, 0.5],
            "winner": "candidate",
            "score_delta": 1.2,
            "defect_labels": ["residual_vocal_bleed"],
            "baseline": {"score": -1.0},
            "candidate": {"score": 0.2},
        },
        {
            "segment_id": "segment_0002",
            "window_sec": [0.5, 1.0],
            "winner": "tie",
            "score_delta": 0.0,
            "defect_labels": ["instrumental_muffling"],
            "baseline": {"score": -0.1},
            "candidate": {"score": -0.1},
        },
    ]
    segment_path = root / "segment_comparison.jsonl"
    segment_path.write_text(
        "\n".join(json.dumps(segment, sort_keys=True) for segment in segments) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "mode": "instrumental_diagnostic_comparison_gate",
        "overall": {
            "status": "candidate_proxy_improved",
            "segment_count": len(segments),
        },
        "segment_comparison_jsonl": str(segment_path),
    }
    path = root / "instrumental_diagnostic_comparison_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


class TestBuildDiagnosticABListeningPack(unittest.TestCase):
    def test_preserves_single_clip_subtypes_and_samples_without_gain(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            baseline_path = temp_path / "baseline.flac"
            candidate_path = temp_path / "candidate.flac"
            baseline = np.concatenate(
                [
                    np.full(sample_rate // 2, 0.25, dtype=np.float64),
                    np.full(sample_rate // 2, -0.125, dtype=np.float64),
                ]
            )
            candidate = np.concatenate(
                [
                    np.full(sample_rate // 2, 0.5, dtype=np.float64),
                    np.full(sample_rate // 2, -0.0625, dtype=np.float64),
                ]
            )
            stereo_baseline = np.column_stack([baseline, baseline])
            stereo_candidate = np.column_stack([candidate, candidate])
            _write_flac(baseline_path, stereo_baseline, sample_rate, subtype="PCM_24")
            _write_flac(candidate_path, stereo_candidate, sample_rate, subtype="PCM_16")
            comparison_manifest = _write_comparison_manifest(temp_path)
            output_dir = temp_path / "ab_pack"

            manifest = module.build_ab_listening_pack(
                comparison_manifest_path=comparison_manifest,
                baseline_audio_path=baseline_path,
                candidate_audio_path=candidate_path,
                output_dir=output_dir,
                baseline_label="baseline",
                candidate_label="candidate",
            )

            self.assertEqual(manifest["mode"], "diagnostic_ab_listening_gate_pack_no_gain")
            self.assertFalse(manifest["audio_write_contract"]["gain_applied"])
            self.assertFalse(manifest["audio_write_contract"]["normalization_applied"])
            self.assertFalse(manifest["audio_write_contract"]["limiting_applied"])
            self.assertEqual(manifest["audio_write_contract"]["baseline_source_subtype"], "PCM_24")
            self.assertEqual(manifest["audio_write_contract"]["candidate_source_subtype"], "PCM_16")
            self.assertEqual(manifest["audio_write_contract"]["single_clip_subtype_policy"], "preserve_source_subtype")
            self.assertEqual(manifest["audio_write_contract"]["max_A_roundtrip_abs_diff"], 0.0)
            self.assertEqual(manifest["audio_write_contract"]["max_B_roundtrip_abs_diff"], 0.0)
            self.assertFalse(manifest["delivery_contract"]["delivery_allowed"])

            first = manifest["clips"][0]
            a_info = sf.info(first["paths"]["A_baseline"])
            b_info = sf.info(first["paths"]["B_candidate"])
            self.assertEqual(a_info.subtype, "PCM_24")
            self.assertEqual(b_info.subtype, "PCM_16")
            a_audio, _ = sf.read(first["paths"]["A_baseline"], always_2d=True, dtype="float64")
            source_a, _ = sf.read(
                str(baseline_path),
                start=0,
                frames=sample_rate // 2,
                always_2d=True,
                dtype="float64",
            )
            b_audio, _ = sf.read(first["paths"]["B_candidate"], always_2d=True, dtype="float64")
            source_b, _ = sf.read(
                str(candidate_path),
                start=0,
                frames=sample_rate // 2,
                always_2d=True,
                dtype="float64",
            )
            np.testing.assert_array_equal(a_audio, source_a)
            np.testing.assert_array_equal(b_audio, source_b)

    def test_cli_writes_no_gain_manifest(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            baseline_path = temp_path / "baseline.flac"
            candidate_path = temp_path / "candidate.flac"
            _write_flac(baseline_path, np.full((sample_rate, 2), 0.25), sample_rate)
            _write_flac(candidate_path, np.full((sample_rate, 2), 0.5), sample_rate)
            comparison_manifest = _write_comparison_manifest(temp_path)
            output_dir = temp_path / "ab_pack"

            exit_code = module.main(
                [
                    str(comparison_manifest),
                    str(baseline_path),
                    str(candidate_path),
                    str(output_dir),
                    "--baseline-label",
                    "baseline",
                    "--candidate-label",
                    "candidate",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "diagnostic_ab_listening_manifest.json").exists())
