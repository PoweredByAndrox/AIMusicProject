import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.melody_alignment import compare_tracks_by_melody_alignment


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "track_14054_14056_melody_alignment.json"


class TestTrack1405414056MelodyAlignment(unittest.TestCase):
    def test_pair_alignment_matches_sequence_level_baseline(self):
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        track_a_path = Path(fixture["track_a_path"])
        track_b_path = Path(fixture["track_b_path"])
        if not track_a_path.exists():
            raise unittest.SkipTest(f"Track fixture audio missing: {track_a_path}")
        if not track_b_path.exists():
            raise unittest.SkipTest(f"Track fixture audio missing: {track_b_path}")

        pair = compare_tracks_by_melody_alignment(track_a_path, track_b_path, device="cuda")
        identity = compare_tracks_by_melody_alignment(track_a_path, track_a_path, device="cuda")

        self.assertEqual(pair["separator_model"], fixture["separator_model"])
        self.assertEqual(pair["frame_hop_sec"], fixture["frame_hop_sec"])

        min_similarity, max_similarity = fixture["pair_expectations"]["overall_melodic_similarity_range"]
        self.assertTrue(
            min_similarity <= pair["overall_melodic_similarity"] <= max_similarity,
            f"pair overall={pair['overall_melodic_similarity']} outside [{min_similarity}, {max_similarity}]",
        )
        self.assertEqual(
            identity["overall_melodic_similarity"],
            fixture["self_expectations"]["overall_melodic_similarity_exact"],
        )
        self.assertGreater(identity["overall_melodic_similarity"], pair["overall_melodic_similarity"])

        for stem_name, expectation in fixture["pair_expectations"]["stems"].items():
            actual = pair["stems"][stem_name]
            self.assertEqual(actual["melodic_source_mode"], expectation["melodic_source_mode"])
            self.assertGreaterEqual(actual["event_count_a"], expectation["event_count_a_min"])
            self.assertGreaterEqual(actual["event_count_b"], expectation["event_count_b_min"])
            self.assertGreaterEqual(len(actual["melodic_event_chain_a"]), expectation["melodic_event_chain_count_a_min"])
            self.assertGreaterEqual(len(actual["melodic_event_chain_b"]), expectation["melodic_event_chain_count_b_min"])
            self.assertGreaterEqual(len(actual["melodic_phrases_a"]), expectation["phrase_count_a_min"])
            self.assertGreaterEqual(len(actual["melodic_phrases_b"]), expectation["phrase_count_b_min"])
            self.assertEqual(actual["best_transposition_semitones"], expectation["best_transposition_semitones"])

            min_pitch, max_pitch = expectation["pitch_class_dtw_similarity_range"]
            self.assertTrue(
                min_pitch <= actual["pitch_class_dtw_similarity"] <= max_pitch,
                f"stem={stem_name} pitch_class_dtw_similarity={actual['pitch_class_dtw_similarity']} "
                f"outside [{min_pitch}, {max_pitch}]",
            )

            min_note, max_note = expectation["note_name_dtw_similarity_range"]
            self.assertTrue(
                min_note <= actual["note_name_dtw_similarity"] <= max_note,
                f"stem={stem_name} note_name_dtw_similarity={actual['note_name_dtw_similarity']} "
                f"outside [{min_note}, {max_note}]",
            )

            min_combined, max_combined = expectation["combined_similarity_range"]
            self.assertTrue(
                min_combined <= actual["combined_similarity"] <= max_combined,
                f"stem={stem_name} combined_similarity={actual['combined_similarity']} "
                f"outside [{min_combined}, {max_combined}]",
            )

            min_beat_pc, max_beat_pc = expectation["beat_sync_pitch_class_similarity_range"]
            self.assertTrue(
                min_beat_pc <= actual["beat_sync_pitch_class_similarity"] <= max_beat_pc,
                f"stem={stem_name} beat_sync_pitch_class_similarity={actual['beat_sync_pitch_class_similarity']} "
                f"outside [{min_beat_pc}, {max_beat_pc}]",
            )

            min_beat_note, max_beat_note = expectation["beat_sync_note_profile_similarity_range"]
            self.assertTrue(
                min_beat_note <= actual["beat_sync_note_profile_similarity"] <= max_beat_note,
                f"stem={stem_name} beat_sync_note_profile_similarity={actual['beat_sync_note_profile_similarity']} "
                f"outside [{min_beat_note}, {max_beat_note}]",
            )

            min_beat_combined, max_beat_combined = expectation["beat_sync_combined_similarity_range"]
            self.assertTrue(
                min_beat_combined <= actual["beat_sync_combined_similarity"] <= max_beat_combined,
                f"stem={stem_name} beat_sync_combined_similarity={actual['beat_sync_combined_similarity']} "
                f"outside [{min_beat_combined}, {max_beat_combined}]",
            )

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

            for note_name in expectation["shared_top_note_classes_contains"]:
                self.assertIn(note_name, actual["shared_top_note_classes"])

        for stem_name, expectation in fixture["self_expectations"]["stems"].items():
            actual = identity["stems"][stem_name]
            self.assertEqual(actual["pitch_class_dtw_similarity"], expectation["pitch_class_dtw_similarity_exact"])
            self.assertEqual(actual["note_name_dtw_similarity"], expectation["note_name_dtw_similarity_exact"])
            self.assertEqual(actual["combined_similarity"], expectation["combined_similarity_exact"])
            self.assertEqual(actual["beat_sync_pitch_class_similarity"], expectation["beat_sync_pitch_class_similarity_exact"])
            self.assertEqual(actual["beat_sync_note_profile_similarity"], expectation["beat_sync_note_profile_similarity_exact"])
            self.assertEqual(actual["beat_sync_combined_similarity"], expectation["beat_sync_combined_similarity_exact"])
            self.assertEqual(actual["contour_similarity"], expectation["contour_similarity_exact"])
            self.assertEqual(actual["structural_similarity"], expectation["structural_similarity_exact"])
            self.assertEqual(actual["melodic_layer_similarity"], expectation["melodic_layer_similarity_exact"])
            self.assertEqual(actual["motif_alignment"]["similarity"], expectation["motif_similarity_exact"])
            self.assertEqual(actual["phrase_alignment"]["similarity"], expectation["phrase_similarity_exact"])
