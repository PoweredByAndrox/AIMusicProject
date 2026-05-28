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
MODULE_PATH = TOOLS_ROOT / "instrumental_diagnostic_comparison_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("instrumental_diagnostic_comparison_gate", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_audio(path: Path, values: np.ndarray, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), values.astype(np.float32), sample_rate)


def _write_dataset(root: Path, source_audio_path: Path) -> Path:
    segments = [
        {
            "segment_id": "segment_0000",
            "window_sec": [0.0, 0.5],
            "defect_labels": ["instrumental_muffling"],
            "training_targets": ["preserve_instrumental_energy"],
        },
        {
            "segment_id": "segment_0001",
            "window_sec": [0.5, 1.0],
            "defect_labels": ["residual_vocal_bleed"],
            "training_targets": ["suppress_residual_vocal"],
        },
    ]
    segments_path = root / "segments.jsonl"
    segments_path.write_text(
        "\n".join(json.dumps(segment, sort_keys=True) for segment in segments) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "mode": "instrumental_preference_dataset",
        "source_audio_path": str(source_audio_path),
        "segments_jsonl": str(segments_path),
        "segment_count": len(segments),
        "delivery_contract": {
            "delivery_allowed": False,
            "presentation_status": "training_only_not_for_listen_pack",
        },
    }
    manifest_path = root / "instrumental_preference_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _write_combined_defect_dataset(root: Path, source_audio_path: Path) -> Path:
    segments_path = root / "segments_combined.jsonl"
    segments_path.write_text(
        json.dumps(
            {
                "segment_id": "segment_0000",
                "window_sec": [0.0, 1.0],
                "defect_labels": ["residual_vocal_bleed", "instrumental_muffling"],
                "training_targets": ["suppress_residual_vocal", "preserve_instrumental_energy"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "mode": "instrumental_preference_dataset",
        "source_audio_path": str(source_audio_path),
        "segments_jsonl": str(segments_path),
        "segment_count": 1,
        "delivery_contract": {
            "delivery_allowed": False,
            "presentation_status": "training_only_not_for_listen_pack",
        },
    }
    manifest_path = root / "instrumental_preference_dataset_manifest_combined.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


class TestInstrumentalDiagnosticComparisonGate(unittest.TestCase):
    def test_reports_proxy_improvement_without_delivery_claim(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            source_path = temp_path / "source.wav"
            baseline_path = temp_path / "baseline.flac"
            candidate_path = temp_path / "candidate.flac"
            _write_audio(source_path, np.ones(sample_rate, dtype=np.float32), sample_rate)
            _write_audio(
                baseline_path,
                np.concatenate(
                    [
                        np.full(sample_rate // 2, 0.22, dtype=np.float32),
                        np.full(sample_rate // 2, 0.72, dtype=np.float32),
                    ]
                ),
                sample_rate,
            )
            _write_audio(
                candidate_path,
                np.concatenate(
                    [
                        np.full(sample_rate // 2, 0.84, dtype=np.float32),
                        np.full(sample_rate // 2, 0.34, dtype=np.float32),
                    ]
                ),
                sample_rate,
            )
            dataset_manifest = _write_dataset(temp_path, source_path)

            manifest = module.compare_instrumental_candidates(
                source_audio_path=source_path,
                baseline_instrumental_path=baseline_path,
                candidate_instrumental_path=candidate_path,
                output_dir=temp_path / "gate",
                dataset_manifest_path=dataset_manifest,
                baseline_label="melband",
                candidate_label="teacher_calibrated",
            )

            self.assertEqual(manifest["mode"], "instrumental_diagnostic_comparison_gate")
            self.assertEqual(manifest["overall"]["status"], "candidate_proxy_improved")
            self.assertGreater(manifest["overall"]["mean_score_delta"], 0.0)
            self.assertFalse(manifest["delivery_contract"]["delivery_allowed"])
            self.assertEqual(
                manifest["delivery_contract"]["presentation_status"],
                "diagnostic_comparison_not_for_listen_pack",
            )
            self.assertTrue((temp_path / "gate" / "segment_comparison.jsonl").exists())
            self.assertEqual(
                [segment["winner"] for segment in manifest["segments"]],
                ["candidate", "candidate"],
            )

    def test_combined_defects_do_not_reward_over_suppressed_no_reference_candidate(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            source_path = temp_path / "source.wav"
            baseline_path = temp_path / "baseline.flac"
            candidate_path = temp_path / "candidate.flac"
            _write_audio(source_path, np.ones(sample_rate, dtype=np.float32), sample_rate)
            _write_audio(baseline_path, np.full(sample_rate, 0.58, dtype=np.float32), sample_rate)
            _write_audio(candidate_path, np.full(sample_rate, 0.86, dtype=np.float32), sample_rate)
            dataset_manifest = _write_combined_defect_dataset(temp_path, source_path)

            manifest = module.compare_instrumental_candidates(
                source_audio_path=source_path,
                baseline_instrumental_path=baseline_path,
                candidate_instrumental_path=candidate_path,
                output_dir=temp_path / "gate",
                dataset_manifest_path=dataset_manifest,
                baseline_label="over_suppressed",
                candidate_label="preserved",
            )

            self.assertEqual(manifest["overall"]["status"], "candidate_proxy_improved")
            self.assertEqual(manifest["segments"][0]["winner"], "candidate")
            candidate_reasons = manifest["segments"][0]["candidate"]["score_reasons"]
            self.assertIn("target_mode=balanced_no_reference_preservation_first", candidate_reasons)
            self.assertIn("target_rms_ratio=0.860", candidate_reasons)

    def test_reports_mixed_when_candidate_regresses_a_labeled_window(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            source_path = temp_path / "source.wav"
            baseline_path = temp_path / "baseline.flac"
            candidate_path = temp_path / "candidate.flac"
            _write_audio(source_path, np.ones(sample_rate, dtype=np.float32), sample_rate)
            _write_audio(baseline_path, np.full(sample_rate, 0.40, dtype=np.float32), sample_rate)
            _write_audio(candidate_path, np.full(sample_rate, 0.95, dtype=np.float32), sample_rate)
            dataset_manifest = _write_dataset(temp_path, source_path)

            manifest = module.compare_instrumental_candidates(
                source_audio_path=source_path,
                baseline_instrumental_path=baseline_path,
                candidate_instrumental_path=candidate_path,
                output_dir=temp_path / "gate",
                dataset_manifest_path=dataset_manifest,
                baseline_label="melband",
                candidate_label="teacher_calibrated",
            )

            self.assertIn(manifest["overall"]["status"], {"candidate_proxy_regressed", "mixed_or_inconclusive"})
            self.assertGreater(manifest["overall"]["regression_segment_count"], 0)
            self.assertFalse(manifest["delivery_contract"]["delivery_allowed"])

    def test_cli_writes_manifest(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            source_path = temp_path / "source.wav"
            baseline_path = temp_path / "baseline.flac"
            candidate_path = temp_path / "candidate.flac"
            _write_audio(source_path, np.ones(sample_rate, dtype=np.float32), sample_rate)
            _write_audio(baseline_path, np.full(sample_rate, 0.4, dtype=np.float32), sample_rate)
            _write_audio(candidate_path, np.full(sample_rate, 0.5, dtype=np.float32), sample_rate)
            dataset_manifest = _write_dataset(temp_path, source_path)
            output_dir = temp_path / "gate"

            exit_code = module.main(
                [
                    str(source_path),
                    str(baseline_path),
                    str(candidate_path),
                    str(output_dir),
                    "--dataset-manifest",
                    str(dataset_manifest),
                    "--baseline-label",
                    "melband",
                    "--candidate-label",
                    "teacher_calibrated",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "instrumental_diagnostic_comparison_manifest.json").exists())
