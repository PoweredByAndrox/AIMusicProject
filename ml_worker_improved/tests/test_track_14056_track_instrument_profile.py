import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.track_instrument_profile import build_track_instrument_profile


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "track_14056_track_instrument_profile.json"


class TestTrack14056TrackInstrumentProfile(unittest.TestCase):
    def test_track_14056_matches_track_level_instrument_profile_baseline(self):
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        audio_path = Path(fixture["audio_path"])
        if not audio_path.exists():
            raise unittest.SkipTest(f"Track fixture audio missing: {audio_path}")

        actual = build_track_instrument_profile(audio_path, device="cuda")

        self.assertEqual(actual["separator_model"], fixture["separator_model"])
        self.assertEqual(set(actual["stems"].keys()), set(fixture["expected_stems"]))

        global_classes = [item["note"] for item in actual["global_note_inventory"]["top_note_classes"]]
        for note_name in fixture["global_note_classes_contains"]:
            self.assertIn(note_name, global_classes)

        warnings = actual["warnings"]
        for substring in fixture["warning_substrings"]:
            self.assertTrue(
                any(substring in warning for warning in warnings),
                f"missing warning substring={substring!r} warnings={warnings}",
            )

        for stem_name, expectation in fixture["stems"].items():
            actual_stem = actual["stems"][stem_name]
            min_energy, max_energy = expectation["energy_ratio_range"]
            self.assertTrue(
                min_energy <= actual_stem["energy_ratio"] <= max_energy,
                f"stem={stem_name} energy_ratio={actual_stem['energy_ratio']} outside [{min_energy}, {max_energy}]",
            )

            actual_roles = actual_stem["role_labels"]
            self.assertGreater(len(actual_roles), 0)
            self.assertEqual(actual_roles[0]["label"], expectation["top_role"])
            self.assertGreaterEqual(actual_roles[0]["score"], expectation["top_role_score_min"])

            second_role = expectation.get("second_role")
            if second_role is not None:
                self.assertGreaterEqual(len(actual_roles), 2)
                self.assertEqual(actual_roles[1]["label"], second_role)

            note_inventory = actual_stem["note_inventory"]
            self.assertEqual(note_inventory["transcription_mode"], expectation["transcription_mode"])

            note_event_count_exact = expectation.get("note_event_count_exact")
            if note_event_count_exact is not None:
                self.assertEqual(note_inventory["note_event_count"], note_event_count_exact)

            note_event_count_min = expectation.get("note_event_count_min")
            if note_event_count_min is not None:
                self.assertGreaterEqual(note_inventory["note_event_count"], note_event_count_min)

            jamendo_contains = expectation.get("jamendo_contains")
            if jamendo_contains:
                jamendo_labels = [item["label"] for item in actual_stem["instrument_labels"]["jamendo"]]
                for label in jamendo_contains:
                    self.assertIn(label, jamendo_labels)

            note_classes_contains = expectation.get("note_classes_contains")
            if note_classes_contains:
                actual_note_classes = [item["note"] for item in note_inventory["top_note_classes"]]
                for note_name in note_classes_contains:
                    self.assertIn(
                        note_name,
                        actual_note_classes,
                        f"stem={stem_name} top_note_classes={actual_note_classes}",
                    )
