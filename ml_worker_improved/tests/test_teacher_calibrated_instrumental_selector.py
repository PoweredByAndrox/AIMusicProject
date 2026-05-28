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
MODULE_PATH = TOOLS_ROOT / "teacher_calibrated_instrumental_selector.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("teacher_calibrated_instrumental_selector", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_audio(path: Path, values: np.ndarray, sample_rate: int = 8000, subtype: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if subtype is None:
        sf.write(str(path), values.astype(np.float32), sample_rate)
    else:
        sf.write(str(path), values.astype(np.float32), sample_rate, subtype=subtype)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values.astype(np.float64))) + 1e-12))


def _write_candidate_manifest(root: Path, profile: str, audio_path: Path, vocal_path: Path | None = None) -> Path:
    stems = {
        "instrumental": {"path": str(audio_path), "role": "companion"},
    }
    if vocal_path is not None:
        stems["vocals"] = {"path": str(vocal_path), "role": "target"}
    manifest = {
        "schema_version": 1,
        "model_id": profile,
        "model_name": profile,
        "audio": {"sample_rate": 8000, "frames": 8000, "duration_sec": 1.0},
        "stems": stems,
        "contract": {
            "registered_model_candidate": True,
            "fallback_used": False,
            "per_track_hardcode_allowed": False,
        },
    }
    path = root / profile / "stem_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _write_dataset(root: Path, source_audio: Path) -> Path:
    segments_path = root / "segments.jsonl"
    feedback_path = root / "feedback.jsonl"
    segments = [
        {
            "segment_id": "segment_0000",
            "window_sec": [0.0, 0.5],
            "preferred_profile": "melband_roformer_vocals",
            "rating": "good",
            "confidence": 0.9,
            "defect_labels": ["instrumental_muffling"],
            "training_targets": ["preserve_instrumental_energy"],
            "source": "gate_blocker",
            "gate_reasons": ["selected_candidate_still_has_severe_issue"],
            "severity_score": 1.1,
        },
        {
            "segment_id": "segment_0001",
            "window_sec": [0.5, 1.0],
            "preferred_profile": "melband_roformer_vocals",
            "rating": "good",
            "confidence": 0.9,
            "defect_labels": ["residual_vocal_bleed"],
            "training_targets": ["suppress_residual_vocal"],
            "source": "gate_blocker",
            "gate_reasons": ["selected_candidate_still_has_severe_issue"],
            "severity_score": 1.2,
        },
    ]
    segments_path.write_text(
        "\n".join(json.dumps(segment, sort_keys=True) for segment in segments) + "\n",
        encoding="utf-8",
    )
    feedback_path.write_text(
        json.dumps(
            {
                "profile": "melband_roformer_vocals",
                "selected_profile": "melband_roformer_vocals",
                "target_stem": "instrumental",
                "rating": "good",
                "confidence": 0.9,
                "human_verified": True,
                "residual_vocal_bleed": True,
                "too_dry": True,
                "feedback_split": "train",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "mode": "instrumental_preference_dataset",
        "source_audio_path": str(source_audio),
        "preferred_candidate": {
            "profile": "melband_roformer_vocals",
            "instrumental_path": str(root / "melband.flac"),
        },
        "segments_jsonl": str(segments_path),
        "feedback_jsonl": str(feedback_path),
        "segment_count": len(segments),
        "delivery_contract": {
            "delivery_allowed": False,
            "presentation_status": "training_only_not_for_listen_pack",
        },
    }
    manifest_path = root / "instrumental_preference_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


class TestTeacherCalibratedInstrumentalSelector(unittest.TestCase):
    def test_selects_segment_candidates_from_preference_dataset_and_renders_audio(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            source_audio = np.concatenate(
                [
                    np.full(sample_rate // 2, 0.50, dtype=np.float32),
                    np.full(sample_rate // 2, 1.00, dtype=np.float32),
                ]
            )
            preferred_audio = np.concatenate(
                [
                    np.full(sample_rate // 2, 0.20, dtype=np.float32),
                    np.full(sample_rate // 2, 0.70, dtype=np.float32),
                ]
            )
            preserve_audio = np.concatenate(
                [
                    np.full(sample_rate // 2, 0.46, dtype=np.float32),
                    np.full(sample_rate // 2, 0.98, dtype=np.float32),
                ]
            )
            clean_audio = np.concatenate(
                [
                    np.full(sample_rate // 2, 0.16, dtype=np.float32),
                    np.full(sample_rate // 2, 0.30, dtype=np.float32),
                ]
            )
            source_path = temp_path / "source.wav"
            preferred_path = temp_path / "melband.flac"
            preserve_path = temp_path / "preserve.flac"
            clean_path = temp_path / "clean.flac"
            _write_audio(source_path, source_audio, sample_rate)
            _write_audio(preferred_path, preferred_audio, sample_rate, subtype="PCM_24")
            _write_audio(preserve_path, preserve_audio, sample_rate)
            _write_audio(clean_path, clean_audio, sample_rate)

            dataset_manifest = _write_dataset(temp_path, source_path)
            candidate_manifests = [
                _write_candidate_manifest(temp_path, "melband_roformer_vocals", preferred_path),
                _write_candidate_manifest(temp_path, "preserve_candidate", preserve_path),
                _write_candidate_manifest(temp_path, "clean_candidate", clean_path),
            ]
            output_dir = temp_path / "selector"

            manifest = module.build_teacher_calibrated_selector(
                dataset_manifest_path=dataset_manifest,
                output_dir=output_dir,
                candidate_stem_manifest_paths=candidate_manifests,
                render_audio=True,
                crossfade_ms=0.0,
            )

            self.assertEqual(manifest["mode"], "teacher_calibrated_instrumental_selector")
            self.assertEqual(manifest["segment_count"], 2)
            self.assertEqual(
                [segment["selected_profile"] for segment in manifest["segment_selection"]],
                ["preserve_candidate", "clean_candidate"],
            )
            self.assertEqual(
                manifest["delivery_contract"]["presentation_status"],
                "diagnostic_selector_not_for_listen_pack",
            )
            self.assertFalse(manifest["delivery_contract"]["delivery_allowed"])
            self.assertTrue((output_dir / "segment_selection_plan.jsonl").exists())
            self.assertTrue((output_dir / "teacher_calibrated_instrumental.flac").exists())

            rendered, rendered_sr = sf.read(
                str(output_dir / "teacher_calibrated_instrumental.flac"),
                always_2d=False,
                dtype="float32",
            )
            self.assertEqual(rendered_sr, sample_rate)
            self.assertEqual(sf.info(str(output_dir / "teacher_calibrated_instrumental.flac")).subtype, "PCM_24")
            self.assertGreater(_rms(rendered[: sample_rate // 2]), 0.40)
            self.assertLess(_rms(rendered[sample_rate // 2 :]), 0.40)

    def test_segment_human_ab_feedback_overrides_proxy_for_matching_segment(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            source_audio = np.concatenate(
                [
                    np.full(sample_rate // 2, 0.50, dtype=np.float32),
                    np.full(sample_rate // 2, 1.00, dtype=np.float32),
                ]
            )
            preferred_audio = np.concatenate(
                [
                    np.full(sample_rate // 2, 0.20, dtype=np.float32),
                    np.full(sample_rate // 2, 0.70, dtype=np.float32),
                ]
            )
            preserve_audio = np.concatenate(
                [
                    np.full(sample_rate // 2, 0.46, dtype=np.float32),
                    np.full(sample_rate // 2, 0.98, dtype=np.float32),
                ]
            )
            clean_audio = np.concatenate(
                [
                    np.full(sample_rate // 2, 0.16, dtype=np.float32),
                    np.full(sample_rate // 2, 0.30, dtype=np.float32),
                ]
            )
            source_path = temp_path / "source.wav"
            preferred_path = temp_path / "melband.flac"
            preserve_path = temp_path / "preserve.flac"
            clean_path = temp_path / "clean.flac"
            _write_audio(source_path, source_audio, sample_rate)
            _write_audio(preferred_path, preferred_audio, sample_rate)
            _write_audio(preserve_path, preserve_audio, sample_rate)
            _write_audio(clean_path, clean_audio, sample_rate)

            dataset_manifest = _write_dataset(temp_path, source_path)
            dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
            feedback_path = Path(dataset["feedback_jsonl"])
            with feedback_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "mode": "instrumental_segment_ab_feedback",
                            "segment_id": "segment_0000",
                            "window_sec": [0.0, 0.5],
                            "baseline_profile": "melband_roformer_vocals",
                            "candidate_profile": "preserve_candidate",
                            "preferred_profile": "melband_roformer_vocals",
                            "rejected_profile": "preserve_candidate",
                            "profile": "preserve_candidate",
                            "rating": "worse",
                            "human_ab_verdict": "baseline_better",
                            "confidence": 1.0,
                            "human_verified": True,
                            "listener_verified": True,
                            "feedback_split": "train",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

            candidate_manifests = [
                _write_candidate_manifest(temp_path, "melband_roformer_vocals", preferred_path),
                _write_candidate_manifest(temp_path, "preserve_candidate", preserve_path),
                _write_candidate_manifest(temp_path, "clean_candidate", clean_path),
            ]
            output_dir = temp_path / "selector"

            manifest = module.build_teacher_calibrated_selector(
                dataset_manifest_path=dataset_manifest,
                output_dir=output_dir,
                candidate_stem_manifest_paths=candidate_manifests,
                render_audio=False,
            )

            self.assertEqual(
                [segment["selected_profile"] for segment in manifest["segment_selection"]],
                ["melband_roformer_vocals", "clean_candidate"],
            )
            first_reasons = manifest["segment_selection"][0]["candidate_scores"][0]["score_reasons"]
            self.assertTrue(any(reason.startswith("segment_feedback_prior=") for reason in first_reasons))

    def test_balanced_defects_penalize_muffling_and_vocal_stem_leakage(self) -> None:
        module = _load_module()

        labels = ["residual_vocal_bleed", "instrumental_muffling"]
        balanced_score, balanced_reasons = module.score_candidate(
            profile="balanced",
            preferred_profile="baseline",
            labels=labels,
            priors={},
            features={
                "rms_ratio": 0.72,
                "presence_ratio_db_vs_source": -3.0,
                "air_ratio_db_vs_source": -5.0,
                "vocal_stem_corr_abs": 0.05,
                "vocal_projection_db_vs_instrumental": -30.0,
            },
        )
        muffled_score, muffled_reasons = module.score_candidate(
            profile="muffled",
            preferred_profile="baseline",
            labels=labels,
            priors={},
            features={
                "rms_ratio": 0.49,
                "presence_ratio_db_vs_source": -12.0,
                "air_ratio_db_vs_source": -14.0,
                "vocal_stem_corr_abs": 0.04,
                "vocal_projection_db_vs_instrumental": -32.0,
            },
        )
        leaky_score, leaky_reasons = module.score_candidate(
            profile="leaky",
            preferred_profile="baseline",
            labels=labels,
            priors={},
            features={
                "rms_ratio": 0.86,
                "presence_ratio_db_vs_source": -2.0,
                "air_ratio_db_vs_source": -4.0,
                "vocal_stem_corr_abs": 0.26,
                "vocal_projection_db_vs_instrumental": -12.0,
            },
        )

        self.assertGreater(balanced_score, muffled_score)
        self.assertGreater(balanced_score, leaky_score)
        self.assertIn("target_mode=balanced_preserve_when_vocal_leakage_proxy_low", balanced_reasons)
        self.assertIn("target_rms_ratio=0.860", balanced_reasons)
        self.assertIn("target_mode=balanced_suppress_when_vocal_leakage_proxy_present", leaky_reasons)
        self.assertIn("target_rms_ratio=0.720", leaky_reasons)
        self.assertTrue(any(reason.startswith("presence_loss_penalty=") for reason in muffled_reasons))
        self.assertTrue(any(reason.startswith("vocal_stem_corr_penalty=") for reason in leaky_reasons))
        self.assertTrue(any(reason.startswith("vocal_projection_penalty=") for reason in leaky_reasons))

    def test_segment_features_include_optional_vocal_stem_leakage_proxy(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            t = np.arange(sample_rate, dtype=np.float32) / sample_rate
            instrumental = 0.6 * np.sin(2.0 * np.pi * 700.0 * t)
            vocal = 0.5 * np.sin(2.0 * np.pi * 220.0 * t)
            source = instrumental + vocal
            candidate = instrumental + 0.35 * vocal

            source_path = temp_path / "source.wav"
            candidate_path = temp_path / "candidate.flac"
            vocal_path = temp_path / "vocals.flac"
            _write_audio(source_path, source, sample_rate)
            _write_audio(candidate_path, candidate, sample_rate)
            _write_audio(vocal_path, vocal, sample_rate)

            features = module.segment_features(source_path, candidate_path, [0.0, 1.0], vocal_path)

            self.assertIn("vocal_stem_corr_abs", features)
            self.assertIn("vocal_projection_db_vs_instrumental", features)
            self.assertGreater(features["vocal_stem_corr_abs"], 0.20)
            self.assertGreater(features["vocal_projection_db_vs_instrumental"], -22.0)

    def test_cli_writes_selector_manifest(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sample_rate = 8000
            source_path = temp_path / "source.wav"
            preferred_path = temp_path / "melband.flac"
            _write_audio(source_path, np.full(sample_rate, 0.5, dtype=np.float32), sample_rate)
            _write_audio(preferred_path, np.full(sample_rate, 0.3, dtype=np.float32), sample_rate)
            dataset_manifest = _write_dataset(temp_path, source_path)
            candidate_manifest = _write_candidate_manifest(temp_path, "melband_roformer_vocals", preferred_path)
            output_dir = temp_path / "selector"

            exit_code = module.main(
                [
                    str(dataset_manifest),
                    str(output_dir),
                    "--candidate-stem-manifest",
                    str(candidate_manifest),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "teacher_calibrated_selector_manifest.json").exists())
