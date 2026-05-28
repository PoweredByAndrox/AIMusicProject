import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.melody_alignment import compare_tracks_by_melody_alignment


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "track_14084_14088_window_melody_alignment.json"


class TestTrack1408414088WindowMelodyAlignment(unittest.TestCase):
    def test_window_alignment_matches_reference_pair(self):
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        track_a_path = Path(fixture["track_a_path"])
        track_b_path = Path(fixture["track_b_path"])
        if not track_a_path.exists():
            raise unittest.SkipTest(f"Track fixture audio missing: {track_a_path}")
        if not track_b_path.exists():
            raise unittest.SkipTest(f"Track fixture audio missing: {track_b_path}")

        pair = compare_tracks_by_melody_alignment(
            track_a_path,
            track_b_path,
            device="cuda",
            window_a=tuple(fixture["window_a"]),
            window_b=tuple(fixture["window_b"]),
        )
        identity = compare_tracks_by_melody_alignment(
            track_a_path,
            track_a_path,
            device="cuda",
            window_a=tuple(fixture["window_a"]),
            window_b=tuple(fixture["window_a"]),
        )

        self.assertEqual(pair["separator_model"], fixture["separator_model"])
        self.assertEqual(pair["frame_hop_sec"], fixture["frame_hop_sec"])
        self.assertEqual(pair["window_a"]["start_sec"], fixture["window_a"][0])
        self.assertEqual(pair["window_a"]["end_sec"], fixture["window_a"][1])
        self.assertEqual(pair["window_b"]["start_sec"], fixture["window_b"][0])
        self.assertEqual(pair["window_b"]["end_sec"], fixture["window_b"][1])

        min_overall, max_overall = fixture["pair_expectations"]["overall_melodic_similarity_range"]
        self.assertTrue(
            min_overall <= pair["overall_melodic_similarity"] <= max_overall,
            f"overall={pair['overall_melodic_similarity']} outside [{min_overall}, {max_overall}]",
        )
        self.assertEqual(identity["overall_melodic_similarity"], 1.0)
        self.assertGreater(identity["overall_melodic_similarity"], pair["overall_melodic_similarity"])

        for stem_name, expectation in fixture["pair_expectations"]["stems"].items():
            actual = pair["stems"][stem_name]
            self.assertEqual(actual["melodic_source_mode"], expectation["melodic_source_mode"])
            self.assertGreaterEqual(len(actual["melodic_event_chain_a"]), expectation["melodic_event_chain_count_a_min"])
            self.assertGreaterEqual(len(actual["melodic_event_chain_b"]), expectation["melodic_event_chain_count_b_min"])
            self.assertGreaterEqual(len(actual["melodic_phrases_a"]), expectation["phrase_count_a_min"])
            if "phrase_count_a_max" in expectation:
                self.assertLessEqual(len(actual["melodic_phrases_a"]), expectation["phrase_count_a_max"])
            self.assertGreaterEqual(len(actual["melodic_phrases_b"]), expectation["phrase_count_b_min"])
            if "beat_sync_pitch_class_similarity_range" in expectation:
                min_beat_pc, max_beat_pc = expectation["beat_sync_pitch_class_similarity_range"]
                self.assertTrue(
                    min_beat_pc <= actual["beat_sync_pitch_class_similarity"] <= max_beat_pc,
                    f"stem={stem_name} beat_sync_pitch_class_similarity={actual['beat_sync_pitch_class_similarity']} "
                    f"outside [{min_beat_pc}, {max_beat_pc}]",
                )

            if "contour_similarity_range" in expectation:
                min_contour, max_contour = expectation["contour_similarity_range"]
                self.assertTrue(
                    min_contour <= actual["contour_similarity"] <= max_contour,
                    f"stem={stem_name} contour_similarity={actual['contour_similarity']} "
                    f"outside [{min_contour}, {max_contour}]",
                )

            min_structural, max_structural = expectation["structural_similarity_range"]
            self.assertTrue(
                min_structural <= actual["structural_similarity"] <= max_structural,
                f"stem={stem_name} structural_similarity={actual['structural_similarity']} "
                f"outside [{min_structural}, {max_structural}]",
            )

            min_melodic_layer, max_melodic_layer = expectation["melodic_layer_similarity_range"]
            self.assertTrue(
                min_melodic_layer <= actual["melodic_layer_similarity"] <= max_melodic_layer,
                f"stem={stem_name} melodic_layer_similarity={actual['melodic_layer_similarity']} "
                f"outside [{min_melodic_layer}, {max_melodic_layer}]",
            )

            min_motif, max_motif = expectation["motif_similarity_range"]
            self.assertTrue(
                min_motif <= actual["motif_alignment"]["similarity"] <= max_motif,
                f"stem={stem_name} motif_similarity={actual['motif_alignment']['similarity']} "
                f"outside [{min_motif}, {max_motif}]",
            )

            min_phrase, max_phrase = expectation["phrase_similarity_range"]
            self.assertTrue(
                min_phrase <= actual["phrase_alignment"]["similarity"] <= max_phrase,
                f"stem={stem_name} phrase_similarity={actual['phrase_alignment']['similarity']} "
                f"outside [{min_phrase}, {max_phrase}]",
            )

            self.assertEqual(identity["stems"][stem_name]["phrase_alignment"]["similarity"], 1.0)

            for note_name in expectation["shared_top_note_classes_contains"]:
                self.assertIn(note_name, actual["shared_top_note_classes"])
