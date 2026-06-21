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
MODULE_PATH = TOOLS_ROOT / "instrumental_self_judge_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("instrumental_self_judge_gate", MODULE_PATH)
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
    segments_path = root / "segments.jsonl"
    segments = [
        {
            "segment_id": "segment_0000",
            "window_sec": [0.0, 0.5],
            "defect_labels": ["residual_vocal_bleed", "instrumental_muffling"],
        },
        {
            "segment_id": "segment_0001",
            "window_sec": [0.5, 1.0],
            "defect_labels": ["residual_vocal_bleed", "instrumental_muffling"],
        },
    ]
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


class TestInstrumentalSelfJudgeGate(unittest.TestCase):
    def test_proxy_pass_preserved_candidate_without_delivery_claim(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            source_path = temp_path / "source.wav"
            candidate_path = temp_path / "candidate.flac"
            _write_audio(source_path, np.full(sample_rate, 0.8, dtype=np.float32), sample_rate)
            _write_audio(candidate_path, np.full(sample_rate, 0.72, dtype=np.float32), sample_rate)
            dataset_manifest = _write_dataset(temp_path, source_path)

            manifest = module.judge_instrumental(
                source_audio_path=source_path,
                candidate_instrumental_path=candidate_path,
                output_dir=temp_path / "judge",
                dataset_manifest_path=dataset_manifest,
                candidate_label="candidate",
            )

            self.assertEqual(manifest["mode"], "instrumental_self_judge_gate")
            self.assertEqual(manifest["overall"]["status"], "proxy_pass")
            self.assertEqual(manifest["overall"]["blocker_count"], 0)
            self.assertFalse(manifest["delivery_contract"]["delivery_allowed"])
            self.assertEqual(
                manifest["delivery_contract"]["presentation_status"],
                "diagnostic_self_judge_not_for_listen_pack",
            )
            self.assertTrue((temp_path / "judge" / "instrumental_self_judge_segments.jsonl").exists())

    def test_blocks_muffling_and_vocal_projection_proxy(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            t = np.arange(sample_rate, dtype=np.float32) / sample_rate
            instrumental = 0.55 * np.sin(2.0 * np.pi * 700.0 * t)
            vocal = 0.40 * np.sin(2.0 * np.pi * 220.0 * t)
            source = instrumental + vocal
            candidate = np.concatenate(
                [
                    0.18 * source[: sample_rate // 2],
                    instrumental[sample_rate // 2 :] + 0.45 * vocal[sample_rate // 2 :],
                ]
            )
            source_path = temp_path / "source.wav"
            candidate_path = temp_path / "candidate.flac"
            vocal_path = temp_path / "vocals.flac"
            _write_audio(source_path, source, sample_rate)
            _write_audio(candidate_path, candidate, sample_rate)
            _write_audio(vocal_path, vocal, sample_rate)
            dataset_manifest = _write_dataset(temp_path, source_path)

            manifest = module.judge_instrumental(
                source_audio_path=source_path,
                candidate_instrumental_path=candidate_path,
                output_dir=temp_path / "judge",
                candidate_vocal_path=vocal_path,
                dataset_manifest_path=dataset_manifest,
                candidate_label="candidate",
            )

            self.assertEqual(manifest["overall"]["status"], "proxy_blocked")
            self.assertEqual(manifest["overall"]["blocker_count"], 2)
            issue_counts = manifest["overall"]["issue_counts"]
            self.assertGreater(issue_counts.get("low_rms_muffling_proxy", 0), 0)
            self.assertGreater(issue_counts.get("vocal_projection_proxy", 0), 0)

    def test_resamples_candidate_and_vocal_windows_to_source_rate(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_rate = 8000
            candidate_rate = 11025
            source_path = temp_path / "source.wav"
            candidate_path = temp_path / "candidate.flac"
            vocal_path = temp_path / "vocals.flac"
            _write_audio(source_path, np.full(source_rate, 0.8, dtype=np.float32), source_rate)
            _write_audio(candidate_path, np.full(candidate_rate, 0.72, dtype=np.float32), candidate_rate)
            _write_audio(vocal_path, np.full(candidate_rate, 0.05, dtype=np.float32), candidate_rate)

            manifest = module.judge_instrumental(
                source_audio_path=source_path,
                candidate_instrumental_path=candidate_path,
                output_dir=temp_path / "judge",
                candidate_vocal_path=vocal_path,
                candidate_label="candidate",
                segment_sec=0.5,
            )

            self.assertEqual(manifest["overall"]["segment_count"], 2)
            self.assertTrue((temp_path / "judge" / "instrumental_self_judge_manifest.json").exists())

    def test_can_write_short_output_names(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            source_path = temp_path / "source.wav"
            candidate_path = temp_path / "candidate.flac"
            _write_audio(source_path, np.full(sample_rate, 0.8, dtype=np.float32), sample_rate)
            _write_audio(candidate_path, np.full(sample_rate, 0.72, dtype=np.float32), sample_rate)
            output_dir = temp_path / "judge"

            manifest = module.judge_instrumental(
                source_audio_path=source_path,
                candidate_instrumental_path=candidate_path,
                output_dir=output_dir,
                candidate_label="candidate",
                segment_sec=0.5,
                short_output_names=True,
            )

            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "rows.jsonl").exists())
            self.assertTrue((output_dir / "progress.jsonl").exists())
            self.assertFalse((output_dir / "instrumental_self_judge_manifest.json").exists())
            self.assertFalse((output_dir / "instrumental_self_judge_segments.jsonl").exists())
            self.assertEqual(manifest["manifest_path"], str(output_dir / "manifest.json"))
            self.assertEqual(manifest["segments_jsonl"], str(output_dir / "rows.jsonl"))
            self.assertTrue(manifest["short_output_names"])
            self.assertEqual(manifest["artifact_name_policy"]["path_segment_limit"], 32)

    def test_blocks_full_mix_passthrough_even_when_candidate_is_slightly_quieter(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            t = np.arange(sample_rate, dtype=np.float32) / sample_rate
            source = (
                0.45 * np.sin(2.0 * np.pi * 220.0 * t)
                + 0.25 * np.sin(2.0 * np.pi * 880.0 * t)
            )
            candidate = 0.94 * source
            source_path = temp_path / "source.wav"
            candidate_path = temp_path / "candidate.flac"
            _write_audio(source_path, source, sample_rate)
            _write_audio(candidate_path, candidate, sample_rate)

            manifest = module.judge_instrumental(
                source_audio_path=source_path,
                candidate_instrumental_path=candidate_path,
                output_dir=temp_path / "judge",
                candidate_label="near_full_mix_passthrough",
                segment_sec=0.5,
            )

            self.assertEqual(manifest["overall"]["status"], "proxy_blocked")
            self.assertGreater(manifest["overall"]["issue_counts"].get("full_mix_like_risk_proxy", 0), 0)
            first_segment = manifest["segments"][0]
            self.assertGreater(first_segment["metrics"]["source_candidate_corr_abs"], 0.98)
            self.assertLess(first_segment["metrics"]["source_candidate_delta_rms_ratio"], 0.08)

    def test_high_mix_similarity_without_vocal_evidence_is_not_full_mix_blocker(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            t = np.arange(sample_rate, dtype=np.float32) / sample_rate
            music = 0.55 * np.sin(2.0 * np.pi * 110.0 * t)
            vocal = 0.35 * np.sin(2.0 * np.pi * 880.0 * t)
            source = music + 0.015 * vocal
            candidate = music
            source_path = temp_path / "source.wav"
            candidate_path = temp_path / "candidate.flac"
            vocal_path = temp_path / "vocals.flac"
            _write_audio(source_path, source, sample_rate)
            _write_audio(candidate_path, candidate, sample_rate)
            _write_audio(vocal_path, vocal, sample_rate)

            manifest = module.judge_instrumental(
                source_audio_path=source_path,
                candidate_instrumental_path=candidate_path,
                output_dir=temp_path / "judge",
                candidate_vocal_path=vocal_path,
                candidate_label="music_dense_instrumental",
                segment_sec=0.5,
            )

            self.assertEqual(manifest["overall"]["status"], "proxy_pass")
            first_segment = manifest["segments"][0]
            self.assertGreater(first_segment["metrics"]["source_candidate_corr_abs"], 0.98)
            self.assertLess(first_segment["metrics"]["source_candidate_delta_rms_ratio"], 0.08)
            self.assertNotIn("full_mix_like_risk_proxy", first_segment["issues"])

    def test_high_rms_ratio_still_blocks_even_without_vocal_evidence(self) -> None:
        module = _load_module()

        issues, severities = module.classify_issues(
            {
                "rms_ratio": 1.08,
                "presence_ratio_db_vs_source": 0.0,
                "air_ratio_db_vs_source": 0.0,
                "source_candidate_corr_abs": 0.8,
                "source_candidate_delta_rms_ratio": 0.2,
                "vocal_stem_corr_abs": 0.01,
                "vocal_projection_db_vs_instrumental": -40.0,
            }
        )

        self.assertIn("full_mix_like_risk_proxy", issues)
        self.assertGreater(severities["full_mix_like_risk"], 0.0)

    def test_uses_selector_manifest_vocal_features_for_rendered_selector_audio(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            source_path = temp_path / "source.wav"
            candidate_path = temp_path / "candidate.flac"
            _write_audio(source_path, np.full(sample_rate, 0.8, dtype=np.float32), sample_rate)
            _write_audio(candidate_path, np.full(sample_rate, 0.72, dtype=np.float32), sample_rate)
            dataset_manifest = _write_dataset(temp_path, source_path)
            selector_manifest_path = temp_path / "selector_manifest.json"
            selector_manifest_path.write_text(
                json.dumps(
                    {
                        "mode": "teacher_calibrated_instrumental_selector",
                        "segment_selection": [
                            {
                                "segment_id": "segment_0000",
                                "selected_profile": "selected",
                                "candidate_scores": [
                                    {
                                        "profile": "selected",
                                        "features": {
                                            "vocal_stem_corr_abs": 0.26,
                                            "vocal_projection_db_vs_instrumental": -12.0,
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = module.judge_instrumental(
                source_audio_path=source_path,
                candidate_instrumental_path=candidate_path,
                output_dir=temp_path / "judge",
                dataset_manifest_path=dataset_manifest,
                selector_manifest_path=selector_manifest_path,
                candidate_label="rendered_selector",
            )

            first_segment = manifest["segments"][0]
            self.assertIn("vocal_stem_corr_proxy", first_segment["issues"])
            self.assertIn("vocal_projection_proxy", first_segment["issues"])
            self.assertEqual(manifest["overall"]["status"], "proxy_blocked")

    def test_cli_writes_manifest(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            source_path = temp_path / "source.wav"
            candidate_path = temp_path / "candidate.flac"
            _write_audio(source_path, np.full(sample_rate, 0.8, dtype=np.float32), sample_rate)
            _write_audio(candidate_path, np.full(sample_rate, 0.72, dtype=np.float32), sample_rate)
            output_dir = temp_path / "judge"

            exit_code = module.main([str(source_path), str(candidate_path), str(output_dir)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "instrumental_self_judge_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
