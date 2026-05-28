from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_instrumental_preference_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_instrumental_preference_dataset", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_preferred_stem_manifest(root: Path, source_audio: Path) -> Path:
    candidate_dir = root / "melband_roformer_vocals"
    candidate_dir.mkdir(parents=True)
    instrumental_path = candidate_dir / "instrumental.flac"
    vocals_path = candidate_dir / "vocals.flac"
    instrumental_path.write_bytes(b"fake instrumental")
    vocals_path.write_bytes(b"fake vocals")
    stem_manifest = {
        "schema_version": 1,
        "model_id": "melband_roformer_vocals",
        "model_name": "melband_roformer_vocals",
        "source_audio_path": str(source_audio),
        "audio": {"sample_rate": 44100, "frames": 441000, "duration_sec": 10.0},
        "stems": {
            "vocals": {"path": str(vocals_path), "role": "target"},
            "instrumental": {"path": str(instrumental_path), "role": "companion"},
        },
        "contract": {
            "registered_model_candidate": True,
            "fallback_used": False,
            "per_track_hardcode_allowed": False,
        },
    }
    path = candidate_dir / "stem_manifest.json"
    path.write_text(json.dumps(stem_manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _write_gate_manifest(root: Path) -> Path:
    gate_manifest = {
        "schema_version": 1,
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_status": "diagnostic_only",
        "selected_profile": "issue_aware_segment_selection",
        "defect_effectiveness_gate": {
            "status": "blocked_for_listening",
            "blockers": [
                {
                    "reason": "selected_candidate_has_unstable_vocal_consistency",
                    "metrics": {"dropout_frame_ratio": 0.4},
                },
                {
                    "reason": "selected_candidate_still_has_severe_issue",
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "severity_score": 1.2,
                    "labels": ["kick_punch_bleed", "thin_or_overcleaned"],
                },
                {
                    "reason": "selected_candidate_still_has_severe_issue",
                    "start_sec": 5.0,
                    "end_sec": 10.0,
                    "severity_score": 1.4,
                    "labels": ["vocal_tail_loss"],
                },
            ],
        },
        "vocal_next_action_plan": {
            "status": "candidate_space_exhausted",
            "next_action": "escalate_to_new_vocal_route_or_teacher_calibration",
        },
    }
    path = root / "selected_delivery_gate_manifest.json"
    path.write_text(json.dumps(gate_manifest, indent=2) + "\n", encoding="utf-8")
    return path


class TestBuildInstrumentalPreferenceDataset(unittest.TestCase):
    def test_builds_training_only_segments_and_feedback_from_gate_blockers(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_audio = temp_path / "mix.wav"
            source_audio.write_bytes(b"fake mix")
            stem_manifest_path = _write_preferred_stem_manifest(temp_path, source_audio)
            gate_manifest_path = _write_gate_manifest(temp_path)
            output_dir = temp_path / "dataset"

            manifest = module.build_instrumental_preference_dataset(
                source_audio_path=source_audio,
                preferred_stem_manifest_path=stem_manifest_path,
                output_dir=output_dir,
                gate_manifest_path=gate_manifest_path,
                human_defect_labels=("residual_vocal_bleed", "instrumental_muffling"),
                rating="good",
                confidence=0.9,
                feedback_split="train",
            )

            manifest_path = output_dir / "instrumental_preference_dataset_manifest.json"
            segments_path = output_dir / "segments.jsonl"
            feedback_path = output_dir / "feedback.jsonl"
            progress_path = output_dir / "instrumental_preference_progress.jsonl"

            self.assertTrue(manifest_path.exists())
            self.assertTrue(segments_path.exists())
            self.assertTrue(feedback_path.exists())
            self.assertTrue(progress_path.exists())
            self.assertEqual(manifest["mode"], "instrumental_preference_dataset")
            self.assertEqual(manifest["preferred_candidate"]["profile"], "melband_roformer_vocals")
            self.assertEqual(manifest["segment_count"], 2)
            self.assertEqual(
                manifest["delivery_contract"]["presentation_status"],
                "training_only_not_for_listen_pack",
            )
            self.assertFalse(manifest["delivery_contract"]["delivery_allowed"])
            self.assertIn("residual_vocal_bleed", manifest["defect_taxonomy"]["human_reported"])
            self.assertIn("instrumental_muffling", manifest["defect_taxonomy"]["human_reported"])

            segments = [
                json.loads(line)
                for line in segments_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([segment["window_sec"] for segment in segments], [[0.0, 5.0], [5.0, 10.0]])
            self.assertIn("thin_or_overcleaned", segments[0]["defect_labels"])
            self.assertIn("instrumental_muffling", segments[0]["defect_labels"])
            self.assertIn("suppress_residual_vocal", segments[0]["training_targets"])
            self.assertIn("preserve_instrumental_energy", segments[0]["training_targets"])

            feedback_record = json.loads(feedback_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(feedback_record["profile"], "melband_roformer_vocals")
            self.assertEqual(feedback_record["selected_profile"], "melband_roformer_vocals")
            self.assertEqual(feedback_record["target_stem"], "instrumental")
            self.assertEqual(feedback_record["rating"], "good")
            self.assertTrue(feedback_record["human_verified"])
            self.assertTrue(feedback_record["residual_vocal_bleed"])
            self.assertTrue(feedback_record["too_dry"])

    def test_uses_uniform_segments_when_gate_has_no_timed_blockers(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_audio = temp_path / "mix.wav"
            source_audio.write_bytes(b"fake mix")
            stem_manifest_path = _write_preferred_stem_manifest(temp_path, source_audio)
            output_dir = temp_path / "dataset"

            manifest = module.build_instrumental_preference_dataset(
                source_audio_path=source_audio,
                preferred_stem_manifest_path=stem_manifest_path,
                output_dir=output_dir,
                human_defect_labels=("residual_vocal_bleed",),
                rating="better",
                confidence=0.7,
                segment_sec=4.0,
            )

            segments = [
                json.loads(line)
                for line in (output_dir / "segments.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(manifest["segment_count"], 3)
            self.assertEqual([segment["window_sec"] for segment in segments], [[0.0, 4.0], [4.0, 8.0], [8.0, 10.0]])
            self.assertTrue(all(segment["source"] == "uniform_fallback" for segment in segments))

    def test_feedback_flags_include_gate_labels_without_human_labels(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_audio = temp_path / "mix.wav"
            source_audio.write_bytes(b"fake mix")
            stem_manifest_path = _write_preferred_stem_manifest(temp_path, source_audio)
            gate_manifest_path = _write_gate_manifest(temp_path)
            output_dir = temp_path / "dataset"

            module.build_instrumental_preference_dataset(
                source_audio_path=source_audio,
                preferred_stem_manifest_path=stem_manifest_path,
                output_dir=output_dir,
                gate_manifest_path=gate_manifest_path,
                rating="good",
                confidence=0.8,
            )

            feedback_record = json.loads((output_dir / "feedback.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(feedback_record["residual_vocal_bleed"])
            self.assertTrue(feedback_record["too_dry"])

    def test_cli_writes_dataset_manifest(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_audio = temp_path / "mix.wav"
            source_audio.write_bytes(b"fake mix")
            stem_manifest_path = _write_preferred_stem_manifest(temp_path, source_audio)
            output_dir = temp_path / "dataset"

            exit_code = module.main(
                [
                    str(source_audio),
                    str(output_dir),
                    "--preferred-stem-manifest",
                    str(stem_manifest_path),
                    "--defect-label",
                    "residual_vocal_bleed",
                    "--rating",
                    "good",
                    "--confidence",
                    "0.8",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "instrumental_preference_dataset_manifest.json").exists())
