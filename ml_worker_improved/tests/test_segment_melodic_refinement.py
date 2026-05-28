import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.segment_melodic_refinement import refine_segments_with_note_events


def _event(start: float, end: float, note_class: str, confidence: float = 0.92) -> dict:
    midi_by_class = {
        "C": 60,
        "D": 62,
        "E": 64,
        "F": 65,
        "G": 67,
        "A": 69,
        "B": 71,
    }
    return {
        "start": start,
        "end": end,
        "midi": midi_by_class[note_class],
        "note_name": f"{note_class}4",
        "note_class": note_class,
        "confidence": confidence,
    }


class TestSegmentMelodicRefinement(unittest.TestCase):
    def test_merges_short_adjacent_segments_when_proxy_continuity_is_high(self):
        segments = [
            {"start": 0.0, "end": 4.0, "type": "verse", "importance": 2.0, "label_hint": "chorus"},
            {"start": 4.0, "end": 8.0, "type": "drop_bass", "importance": 3.0, "label_hint": "chorus"},
            {"start": 8.0, "end": 12.0, "type": "verse", "importance": 2.0, "label_hint": "chorus"},
        ]
        other_events = [
            _event(0.0, 3.8, "E"),
            _event(0.2, 3.9, "G"),
            _event(4.0, 7.8, "E"),
            _event(4.2, 7.9, "G"),
            _event(8.0, 11.8, "E"),
            _event(8.2, 11.9, "G"),
        ]
        bass_events = [
            _event(0.0, 3.9, "C"),
            _event(4.0, 7.9, "C"),
            _event(8.0, 11.9, "C"),
        ]

        refined, summary = refine_segments_with_note_events(
            segments,
            other_events=other_events,
            bass_events=bass_events,
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["original_segments"], 3)
        self.assertEqual(summary["refined_segments"], 1)
        self.assertEqual(summary["merges_applied"], 2)
        self.assertEqual(refined[0]["start"], 0.0)
        self.assertEqual(refined[0]["end"], 12.0)

    def test_keeps_boundary_when_note_classes_change_hard(self):
        segments = [
            {"start": 0.0, "end": 6.0, "type": "verse", "importance": 2.0, "label_hint": "verse"},
            {"start": 6.0, "end": 12.0, "type": "verse", "importance": 2.0, "label_hint": "verse"},
        ]
        other_events = [
            _event(0.0, 5.8, "E"),
            _event(0.1, 5.9, "G"),
            _event(6.0, 11.8, "C"),
            _event(6.1, 11.9, "D"),
        ]
        bass_events = [
            _event(0.0, 5.9, "A"),
            _event(6.0, 11.9, "F"),
        ]

        refined, summary = refine_segments_with_note_events(
            segments,
            other_events=other_events,
            bass_events=bass_events,
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["original_segments"], 2)
        self.assertEqual(summary["refined_segments"], 2)
        self.assertEqual(summary["merges_applied"], 0)

    def test_splits_long_undersegmented_region_when_internal_pitch_changes_hard(self):
        segments = [
            {"start": 0.0, "end": 52.0, "type": "ambient", "importance": 1.0, "label_hint": "inst"},
            {"start": 52.0, "end": 60.0, "type": "break", "importance": 1.0, "label_hint": "intro"},
        ]
        other_events = []
        bass_events = []
        for start in [14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0]:
            other_events.append(_event(start, start + 0.9, "E"))
            other_events.append(_event(start + 0.1, start + 1.0, "G"))
            bass_events.append(_event(start + 0.05, start + 0.95, "C"))
        for start in [18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0]:
            other_events.append(_event(start, start + 0.9, "C"))
            other_events.append(_event(start + 0.1, start + 1.0, "D"))
            bass_events.append(_event(start + 0.05, start + 0.95, "F"))
        for start in [32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0]:
            other_events.append(_event(start, start + 0.9, "C"))
            other_events.append(_event(start + 0.1, start + 1.0, "D"))
            bass_events.append(_event(start + 0.05, start + 0.95, "F"))
        for start in [36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0]:
            other_events.append(_event(start, start + 0.9, "A"))
            other_events.append(_event(start + 0.1, start + 1.0, "B"))
            bass_events.append(_event(start + 0.05, start + 0.95, "G"))

        refined, summary = refine_segments_with_note_events(
            segments,
            other_events=other_events,
            bass_events=bass_events,
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["original_segments"], 2)
        self.assertGreaterEqual(summary["segments_split"], 1)
        self.assertGreaterEqual(summary["refined_segments"], 3)
        self.assertAlmostEqual(refined[0]["start"], 0.0)
        self.assertAlmostEqual(refined[-1]["end"], 60.0)


if __name__ == "__main__":
    unittest.main()
