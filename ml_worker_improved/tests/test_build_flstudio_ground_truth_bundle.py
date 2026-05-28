import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_flstudio_ground_truth_bundle.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_flstudio_ground_truth_bundle", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestBuildFlstudioGroundTruthBundle(unittest.TestCase):
    def test_classifies_fl_stem_names_without_track_specific_rules(self):
        module = _load_module()

        cases = {
            "Square a Saw - Oxygen - 808 hats.wav": "drums",
            "Square a Saw - Oxygen (Tevlo Vocals Resynthesized) - kick.wav": "drums",
            "Square a Saw - Oxygen - top bass line.wav": "bass",
            "Square a Saw - Oxygen - Tevlo Vocals Harmor.wav": "vocals",
            "Square a Saw - Oxygen - piano melody.wav": "piano",
            "Square a Saw - Oxygen - string accents.wav": "strings_orchestra",
            "Square a Saw - Oxygen - white noise sweep.wav": "fx",
            "Square a Saw - Oxygen - lead synth.wav": "melody",
        }

        for file_name, expected_stem in cases.items():
            with self.subTest(file_name=file_name):
                self.assertEqual(module.classify_flstudio_stem_name(file_name), expected_stem)

    def test_builds_verified_bundle_with_hashes_and_mix_drift_report(self):
        module = _load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "fl_export"
            output_dir = root / "bundle"
            input_dir.mkdir()
            sample_rate = 8000
            frame_count = sample_rate
            time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)

            stems = {
                "Track - kick.wav": 0.22 * np.sin(2 * np.pi * 60 * time),
                "Track - 808 hats.wav": 0.04 * np.sin(2 * np.pi * 900 * time),
                "Track - sub bass.wav": 0.18 * np.sin(2 * np.pi * 90 * time),
                "Track - Tevlo Vocals Bounce.wav": 0.11 * np.sin(2 * np.pi * 220 * time),
                "Track - piano melody.wav": 0.07 * np.sin(2 * np.pi * 440 * time),
                "Track - lead synth.wav": 0.05 * np.sin(2 * np.pi * 660 * time),
                "Track - white noise sweep.wav": np.linspace(0.0, 0.04, frame_count, dtype=np.float32),
            }
            for file_name, mono in stems.items():
                stereo = np.column_stack([mono, mono]).astype(np.float32)
                sf.write(input_dir / file_name, stereo, sample_rate, subtype="FLOAT")

            full_mix = np.sum(np.stack(list(stems.values()), axis=0), axis=0)
            source_mix_path = input_dir / "Track.flac"
            sf.write(source_mix_path, np.column_stack([full_mix, full_mix]), sample_rate, subtype="PCM_24")

            manifest = module.build_flstudio_ground_truth_bundle(
                source_mix_path=source_mix_path,
                stem_dir=input_dir,
                output_dir=output_dir,
                track_name="Track",
            )

            manifest_path = output_dir / "ground_truth_manifest.json"
            self.assertTrue(manifest_path.exists())
            reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(reloaded["ground_truth_contract"]["verified"])
            self.assertEqual(reloaded["ground_truth_contract"]["verification_method"], "flstudio_split_mixer_tracks_import")
            self.assertIn("sha256", reloaded["stems"]["drums"])
            self.assertIn("sha256", reloaded["stems"]["instrumental"])
            self.assertTrue((output_dir / "stems" / "vocals.flac").exists())
            self.assertTrue((output_dir / "stems" / "instrumental.flac").exists())
            self.assertTrue((output_dir / "reference_sum_mix.flac").exists())
            self.assertIn("reference_sum_mix", reloaded)
            self.assertEqual(reloaded["mix"]["role"], "authoritative_full_mix_input")
            self.assertEqual(reloaded["reference_sum_mix"]["role"], "diagnostic_imported_stem_sum")
            self.assertTrue(reloaded["reference_sum_mix"]["not_authoritative_mix"])
            self.assertEqual(reloaded["stems"]["drums"]["reference_role"], "individual_stem_reference_target")
            self.assertIn("individual reference targets", " ".join(reloaded["ground_truth_contract"]["notes"]))
            self.assertLess(reloaded["reference_sum_check"]["sum_minus_source_rms_dbfs"], -80.0)
            self.assertEqual(reloaded["full_mix_reconstruction_contract"]["status"], "accepted")
            self.assertTrue(reloaded["full_mix_reconstruction_contract"]["can_claim_source_mix_1_to_1"])
            self.assertEqual(manifest["source_file_count"], len(stems))

    def test_records_report_alignment_metadata_for_validation_pack(self):
        module = _load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "fl_export"
            output_dir = root / "bundle"
            input_dir.mkdir()
            sample_rate = 8000
            frame_count = sample_rate // 4
            time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)

            vocal = 0.12 * np.sin(2 * np.pi * 440 * time)
            kick = np.zeros(frame_count, dtype=np.float32)
            kick[::800] = 0.4
            source_mix = vocal + kick
            source_mix_path = input_dir / "Song.flac"
            sf.write(source_mix_path, np.column_stack([source_mix, source_mix]), sample_rate, subtype="PCM_24")
            sf.write(input_dir / "Song - Lead Vocal.wav", np.column_stack([vocal, vocal]), sample_rate, subtype="FLOAT")
            sf.write(input_dir / "Song - Kick.wav", np.column_stack([kick, kick]), sample_rate, subtype="FLOAT")

            manifest = module.build_flstudio_ground_truth_bundle(
                source_mix_path=source_mix_path,
                stem_dir=input_dir,
                output_dir=output_dir,
                track_name="Song",
                bundle_kind="hard_case",
                master_chain_status="mastered_export",
                rights_status="licensed",
                license_notes="internal validation only",
                validation_pack_tags=("hard_case", "commercial_mastered"),
                hard_case_tags=("dense_transients", "quiet_vocal"),
            )

            report_contract = manifest["report_alignment_contract"]
            self.assertEqual(report_contract["source_document"], "deep-research-report.md")
            self.assertEqual(report_contract["bundle_kind"], "hard_case")
            self.assertEqual(report_contract["master_chain_status"], "mastered_export")
            self.assertEqual(report_contract["rights_status"], "licensed")
            self.assertEqual(report_contract["license_notes"], "internal validation only")
            self.assertEqual(report_contract["validation_pack_tags"], ["hard_case", "commercial_mastered"])
            self.assertEqual(report_contract["hard_case_tags"], ["dense_transients", "quiet_vocal"])
            self.assertEqual(
                report_contract["required_qc"],
                [
                    "sample_rate_consistency",
                    "sample_accurate_alignment",
                    "loudness_and_peak_summary",
                    "tail_integrity_summary",
                    "full_mix_reconstruction_contract",
                ],
            )

    def test_builds_drums_plus_percussive_fx_aux_without_changing_strict_drums(self):
        module = _load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "fl_export"
            output_dir = root / "bundle"
            input_dir.mkdir()
            sample_rate = 8000
            frame_count = sample_rate
            time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)

            kick = 0.22 * np.sin(2 * np.pi * 60 * time)
            clap_riser = np.zeros(frame_count, dtype=np.float32)
            clap_riser[2000:2128] = 0.30 * np.hanning(128).astype(np.float32)
            white_noise_sweep = np.linspace(0.0, 0.04, frame_count, dtype=np.float32)
            vocal = 0.11 * np.sin(2 * np.pi * 220 * time)
            stems = {
                "Track - kick.wav": kick,
                "Track - clap riser.wav": clap_riser,
                "Track - white noise sweep.wav": white_noise_sweep,
                "Track - vocal.wav": vocal,
            }
            for file_name, mono in stems.items():
                sf.write(input_dir / file_name, np.column_stack([mono, mono]), sample_rate, subtype="FLOAT")

            source_mix_path = input_dir / "Track.flac"
            full_mix = np.sum(np.stack(list(stems.values()), axis=0), axis=0)
            sf.write(source_mix_path, np.column_stack([full_mix, full_mix]), sample_rate, subtype="PCM_24")

            manifest = module.build_flstudio_ground_truth_bundle(
                source_mix_path=source_mix_path,
                stem_dir=input_dir,
                output_dir=output_dir,
                track_name="Track",
            )
            reloaded = json.loads((output_dir / "ground_truth_manifest.json").read_text(encoding="utf-8"))

            assignments = {entry["name"]: entry["assigned_stem"] for entry in reloaded["source_files"]}
            self.assertEqual(assignments["Track - clap riser.wav"], "fx")
            self.assertEqual(assignments["Track - white noise sweep.wav"], "fx")
            self.assertIn("drums_plus_percussive_fx_aux", reloaded["stems"])
            self.assertTrue(reloaded["stems"]["drums_plus_percussive_fx_aux"]["derived"])
            self.assertEqual(
                reloaded["stems"]["drums_plus_percussive_fx_aux"]["reference_role"],
                "composite_reference_target",
            )
            self.assertEqual(
                reloaded["target_definitions"]["drums_plus_percussive_fx_aux"]["child_stems"],
                ["drums", "fx"],
            )
            self.assertEqual(
                reloaded["target_definitions"]["drums_plus_percussive_fx_aux"]["aux_source_files"],
                ["Track - clap riser.wav"],
            )

            strict_drums, _ = sf.read(output_dir / "stems" / "drums.flac", always_2d=True, dtype="float32")
            composite_drums, _ = sf.read(
                output_dir / "stems" / "drums_plus_percussive_fx_aux.flac",
                always_2d=True,
                dtype="float32",
            )
            expected_strict = np.column_stack([kick, kick]).astype(np.float32)
            expected_composite = np.column_stack([kick + clap_riser, kick + clap_riser]).astype(np.float32)
            self.assertLess(np.max(np.abs(strict_drums - expected_strict)), 2.0e-4)
            self.assertLess(np.max(np.abs(composite_drums - expected_composite)), 2.0e-4)
            self.assertIn("drums_plus_percussive_fx_aux", manifest["stems"])

    def test_marks_source_mix_reconstruction_rejected_when_full_export_differs_from_stem_sum(self):
        module = _load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "fl_export"
            output_dir = root / "bundle"
            input_dir.mkdir()
            sample_rate = 8000
            frame_count = sample_rate
            time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)

            kick = 0.22 * np.sin(2 * np.pi * 60 * time)
            vocal = 0.11 * np.sin(2 * np.pi * 220 * time)
            sf.write(input_dir / "Track - kick.wav", np.column_stack([kick, kick]), sample_rate, subtype="FLOAT")
            sf.write(
                input_dir / "Track - Tevlo Vocals Bounce.wav",
                np.column_stack([vocal, vocal]),
                sample_rate,
                subtype="FLOAT",
            )

            stem_sum = kick + vocal
            mastered_full_mix = 0.6 * stem_sum
            source_mix_path = input_dir / "Track.flac"
            sf.write(source_mix_path, np.column_stack([mastered_full_mix, mastered_full_mix]), sample_rate, subtype="PCM_24")

            manifest = module.build_flstudio_ground_truth_bundle(
                source_mix_path=source_mix_path,
                stem_dir=input_dir,
                output_dir=output_dir,
                track_name="Track",
            )

            contract = manifest["full_mix_reconstruction_contract"]
            self.assertTrue(manifest["ground_truth_contract"]["verified"])
            self.assertEqual(manifest["ground_truth_contract"]["mode"], "verified_isolated_stems")
            self.assertEqual(manifest["mix"]["role"], "authoritative_full_mix_input")
            self.assertEqual(manifest["reference_sum_mix"]["role"], "diagnostic_imported_stem_sum")
            self.assertTrue(manifest["reference_sum_mix"]["not_authoritative_mix"])
            self.assertEqual(manifest["stems"]["drums"]["reference_role"], "individual_stem_reference_target")
            self.assertEqual(contract["status"], "rejected")
            self.assertFalse(contract["can_claim_source_mix_1_to_1"])
            self.assertIn("residual_rms_above_threshold", contract["failed_check_names"])
            self.assertIn("Source full mix is not verified as the exact sum of imported stems.", contract["problem_summary"])

    def test_rejects_mixed_stem_dir_with_foreign_track_prefix_by_default(self):
        module = _load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "fl_export"
            output_dir = root / "bundle"
            input_dir.mkdir()
            sample_rate = 8000
            frame_count = sample_rate // 4
            time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)

            vocal = 0.11 * np.sin(2 * np.pi * 220 * time)
            unrelated_song = 0.08 * np.sin(2 * np.pi * 330 * time)
            source_mix_path = input_dir / "Track.flac"
            sf.write(source_mix_path, np.column_stack([vocal, vocal]), sample_rate, subtype="PCM_24")
            sf.write(input_dir / "Track - vocal.wav", np.column_stack([vocal, vocal]), sample_rate, subtype="FLOAT")
            sf.write(
                input_dir / "Other Artist - Other Song.flac",
                np.column_stack([unrelated_song, unrelated_song]),
                sample_rate,
                subtype="PCM_24",
            )

            with self.assertRaisesRegex(ValueError, "foreign track prefix"):
                module.build_flstudio_ground_truth_bundle(
                    source_mix_path=source_mix_path,
                    stem_dir=input_dir,
                    output_dir=output_dir,
                    track_name="Track",
                )

            self.assertFalse((output_dir / "ground_truth_manifest.json").exists())

    def test_records_input_scope_warning_when_mixed_stem_dir_is_explicitly_allowed(self):
        module = _load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "fl_export"
            output_dir = root / "bundle"
            input_dir.mkdir()
            sample_rate = 8000
            frame_count = sample_rate // 4
            time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)

            vocal = 0.11 * np.sin(2 * np.pi * 220 * time)
            unrelated_song = 0.08 * np.sin(2 * np.pi * 330 * time)
            source_mix_path = input_dir / "Track.flac"
            sf.write(source_mix_path, np.column_stack([vocal, vocal]), sample_rate, subtype="PCM_24")
            sf.write(input_dir / "Track - vocal.wav", np.column_stack([vocal, vocal]), sample_rate, subtype="FLOAT")
            sf.write(
                input_dir / "Other Artist - Other Song.flac",
                np.column_stack([unrelated_song, unrelated_song]),
                sample_rate,
                subtype="PCM_24",
            )

            manifest = module.build_flstudio_ground_truth_bundle(
                source_mix_path=source_mix_path,
                stem_dir=input_dir,
                output_dir=output_dir,
                track_name="Track",
                allow_mixed_stem_dir=True,
            )

            input_scope_contract = manifest["input_scope_contract"]
            self.assertEqual(input_scope_contract["status"], "accepted_with_warnings")
            self.assertTrue(input_scope_contract["allow_mixed_stem_dir"])
            self.assertEqual(
                input_scope_contract["suspicious_foreign_track_files"][0]["name"],
                "Other Artist - Other Song.flac",
            )
            self.assertEqual(manifest["source_file_count"], 2)

    def test_rejects_near_match_foreign_track_prefix_by_default(self):
        module = _load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "fl_export"
            output_dir = root / "bundle"
            input_dir.mkdir()
            sample_rate = 8000
            frame_count = sample_rate // 4
            time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)

            vocal = 0.11 * np.sin(2 * np.pi * 220 * time)
            source_mix_path = input_dir / "Track.flac"
            sf.write(source_mix_path, np.column_stack([vocal, vocal]), sample_rate, subtype="PCM_24")
            sf.write(input_dir / "Track - vocal.wav", np.column_stack([vocal, vocal]), sample_rate, subtype="FLOAT")
            sf.write(input_dir / "Track 2 - vocal.wav", np.column_stack([vocal, vocal]), sample_rate, subtype="FLOAT")

            with self.assertRaisesRegex(ValueError, "foreign track prefix"):
                module.build_flstudio_ground_truth_bundle(
                    source_mix_path=source_mix_path,
                    stem_dir=input_dir,
                    output_dir=output_dir,
                    track_name="Track",
                )

    def test_rejects_unprefixed_unknown_audio_by_default(self):
        module = _load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "fl_export"
            output_dir = root / "bundle"
            input_dir.mkdir()
            sample_rate = 8000
            frame_count = sample_rate // 4
            time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)

            vocal = 0.11 * np.sin(2 * np.pi * 220 * time)
            preview = 0.08 * np.sin(2 * np.pi * 330 * time)
            source_mix_path = input_dir / "Track.flac"
            sf.write(source_mix_path, np.column_stack([vocal, vocal]), sample_rate, subtype="PCM_24")
            sf.write(input_dir / "Track - vocal.wav", np.column_stack([vocal, vocal]), sample_rate, subtype="FLOAT")
            sf.write(input_dir / "preview.wav", np.column_stack([preview, preview]), sample_rate, subtype="FLOAT")

            with self.assertRaisesRegex(ValueError, "unscoped unknown audio"):
                module.build_flstudio_ground_truth_bundle(
                    source_mix_path=source_mix_path,
                    stem_dir=input_dir,
                    output_dir=output_dir,
                    track_name="Track",
                )

    def test_accepts_standalone_semantic_stem_names_without_track_prefix(self):
        module = _load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "fl_export"
            output_dir = root / "bundle"
            input_dir.mkdir()
            sample_rate = 8000
            frame_count = sample_rate // 4
            time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)

            vocal = 0.11 * np.sin(2 * np.pi * 220 * time)
            kick = 0.22 * np.sin(2 * np.pi * 60 * time)
            source_mix_path = input_dir / "Track.flac"
            sf.write(source_mix_path, np.column_stack([vocal + kick, vocal + kick]), sample_rate, subtype="PCM_24")
            sf.write(input_dir / "vocal.wav", np.column_stack([vocal, vocal]), sample_rate, subtype="FLOAT")
            sf.write(input_dir / "kick.wav", np.column_stack([kick, kick]), sample_rate, subtype="FLOAT")

            manifest = module.build_flstudio_ground_truth_bundle(
                source_mix_path=source_mix_path,
                stem_dir=input_dir,
                output_dir=output_dir,
                track_name="Track",
            )

            self.assertEqual(manifest["input_scope_contract"]["status"], "accepted")
            self.assertEqual(manifest["source_file_count"], 2)

    def test_uses_track_name_to_accept_legitimate_prefix_when_mix_file_is_generic(self):
        module = _load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "fl_export"
            output_dir = root / "bundle"
            input_dir.mkdir()
            sample_rate = 8000
            frame_count = sample_rate // 4
            time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)

            vocal = 0.11 * np.sin(2 * np.pi * 220 * time)
            source_mix_path = input_dir / "master.wav"
            sf.write(source_mix_path, np.column_stack([vocal, vocal]), sample_rate, subtype="FLOAT")
            sf.write(input_dir / "Track - vocal.wav", np.column_stack([vocal, vocal]), sample_rate, subtype="FLOAT")

            manifest = module.build_flstudio_ground_truth_bundle(
                source_mix_path=source_mix_path,
                stem_dir=input_dir,
                output_dir=output_dir,
                track_name="Track",
            )

            self.assertEqual(manifest["input_scope_contract"]["status"], "accepted")

    def test_cli_allow_mixed_stem_dir_records_input_scope_warning(self):
        module = _load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "fl_export"
            output_dir = root / "bundle"
            input_dir.mkdir()
            sample_rate = 8000
            frame_count = sample_rate // 4
            time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)

            vocal = 0.11 * np.sin(2 * np.pi * 220 * time)
            unrelated_song = 0.08 * np.sin(2 * np.pi * 330 * time)
            source_mix_path = input_dir / "Track.flac"
            sf.write(source_mix_path, np.column_stack([vocal, vocal]), sample_rate, subtype="PCM_24")
            sf.write(input_dir / "Track - vocal.wav", np.column_stack([vocal, vocal]), sample_rate, subtype="FLOAT")
            sf.write(
                input_dir / "Other Artist - Other Song.flac",
                np.column_stack([unrelated_song, unrelated_song]),
                sample_rate,
                subtype="PCM_24",
            )

            exit_code = module.main(
                [
                    str(source_mix_path),
                    str(input_dir),
                    str(output_dir),
                    "--track-name",
                    "Track",
                    "--allow-mixed-stem-dir",
                ]
            )

            self.assertEqual(exit_code, 0)
            manifest = json.loads((output_dir / "ground_truth_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["input_scope_contract"]["status"], "accepted_with_warnings")

    def test_cli_require_full_mix_reconstruction_returns_nonzero_for_drifted_full_export(self):
        module = _load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "fl_export"
            output_dir = root / "bundle"
            input_dir.mkdir()
            sample_rate = 8000
            frame_count = sample_rate
            time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)

            kick = 0.22 * np.sin(2 * np.pi * 60 * time)
            vocal = 0.11 * np.sin(2 * np.pi * 220 * time)
            sf.write(input_dir / "Track - kick.wav", np.column_stack([kick, kick]), sample_rate, subtype="FLOAT")
            sf.write(input_dir / "Track - vocal.wav", np.column_stack([vocal, vocal]), sample_rate, subtype="FLOAT")
            source_mix_path = input_dir / "Track.flac"
            sf.write(
                source_mix_path,
                np.column_stack([0.6 * (kick + vocal), 0.6 * (kick + vocal)]),
                sample_rate,
                subtype="PCM_24",
            )

            exit_code = module.main(
                [
                    str(source_mix_path),
                    str(input_dir),
                    str(output_dir),
                    "--track-name",
                    "Track",
                    "--require-full-mix-reconstruction",
                ]
            )

            self.assertEqual(exit_code, 2)
            manifest = json.loads((output_dir / "ground_truth_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["full_mix_reconstruction_contract"]["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
