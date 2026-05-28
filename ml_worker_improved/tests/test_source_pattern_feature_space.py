from __future__ import annotations

import unittest
from unittest.mock import patch

from worker.source_pattern_feature_space import (
    DEFAULT_SEPARATOR_MODEL,
    _profile_change_score,
    compare_window_source_profiles,
    pick_top_source_windows,
)
from worker.stem_separator_defaults import DEFAULT_PATTERN_SEARCH_MODEL, DEFAULT_STEM_EXPORT_MODEL


def _profile(
    *,
    energy: list[float],
    melodic_ratio: list[float],
    chroma: list[float],
    other_pitch: list[float],
    other_interval: list[float],
    other_contour: list[float],
    bass_pitch: list[float],
    bass_interval: list[float],
    other_top: list[str],
    other_chroma: list[str],
    phrase_count: int,
) -> dict:
    return {
        "feature_vectors": {
            "energy": energy,
            "melodic_ratio": melodic_ratio,
            "aggregate_chroma": chroma,
        },
        "note_patterns": {
            "other": {
                "pitch_histogram": other_pitch,
                "interval_histogram": other_interval,
                "contour_histogram": other_contour,
                "phrase_count": phrase_count,
                "top_note_classes": other_top,
            },
            "bass": {
                "pitch_histogram": bass_pitch,
                "interval_histogram": bass_interval,
                "phrase_count": 1,
                "top_note_classes": ["E", "B"],
            },
        },
        "stems": {
            "other": {"top_chroma_notes": other_chroma},
        },
    }


class TestSourcePatternFeatureSpace(unittest.TestCase):
    def test_default_separator_model_matches_pattern_search_baseline(self):
        self.assertEqual(DEFAULT_SEPARATOR_MODEL, DEFAULT_PATTERN_SEARCH_MODEL)
        self.assertNotEqual(DEFAULT_SEPARATOR_MODEL, DEFAULT_STEM_EXPORT_MODEL)

    def test_compare_window_source_profiles_prefers_matching_patterns(self):
        matching_left = _profile(
            energy=[0.7, 0.2, 0.1],
            melodic_ratio=[0.8, 0.1, 0.1],
            chroma=[1.0, 0.0, 0.0, 0.0],
            other_pitch=[1.0, 0.0, 0.0],
            other_interval=[0.8, 0.2, 0.0],
            other_contour=[0.0, 0.2, 0.8],
            bass_pitch=[0.1, 0.9, 0.0],
            bass_interval=[0.9, 0.1, 0.0],
            other_top=["E", "G"],
            other_chroma=["E", "G"],
            phrase_count=4,
        )
        matching_right = _profile(
            energy=[0.68, 0.22, 0.1],
            melodic_ratio=[0.79, 0.11, 0.1],
            chroma=[0.99, 0.01, 0.0, 0.0],
            other_pitch=[0.98, 0.02, 0.0],
            other_interval=[0.75, 0.25, 0.0],
            other_contour=[0.0, 0.15, 0.85],
            bass_pitch=[0.12, 0.88, 0.0],
            bass_interval=[0.88, 0.12, 0.0],
            other_top=["E", "G"],
            other_chroma=["E", "G"],
            phrase_count=5,
        )
        mismatching = _profile(
            energy=[0.15, 0.75, 0.1],
            melodic_ratio=[0.2, 0.7, 0.1],
            chroma=[0.0, 0.0, 1.0, 0.0],
            other_pitch=[0.0, 0.0, 1.0],
            other_interval=[0.0, 0.1, 0.9],
            other_contour=[0.9, 0.1, 0.0],
            bass_pitch=[0.0, 0.2, 0.8],
            bass_interval=[0.0, 0.0, 1.0],
            other_top=["C"],
            other_chroma=["C"],
            phrase_count=1,
        )

        matching_score = compare_window_source_profiles(matching_left, matching_right)
        mismatching_score = compare_window_source_profiles(matching_left, mismatching)

        self.assertGreater(matching_score["overall_similarity"], mismatching_score["overall_similarity"])
        self.assertGreater(matching_score["component_similarities"]["aggregate_chroma_similarity"], 0.9)
        self.assertEqual(matching_score["shared_other_notes"], ["E", "G"])

    @patch("worker.source_pattern_feature_space.extract_window_source_profile")
    @patch("worker.source_pattern_feature_space._track_duration")
    def test_pick_top_source_windows_keeps_non_overlapping_high_scores(self, track_duration_mock, profile_mock):
        track_duration_mock.return_value = 60.0

        def fake_profile(*args, **kwargs):
            start = float(kwargs["window_start"])
            score_table = {
                0.0: {"other_energy": 0.1, "other_melodic": 0.1, "phrase_count": 1, "unique_pitch_classes": 2},
                10.0: {"other_energy": 0.5, "other_melodic": 0.8, "phrase_count": 5, "unique_pitch_classes": 6},
                20.0: {"other_energy": 0.49, "other_melodic": 0.78, "phrase_count": 5, "unique_pitch_classes": 6},
                30.0: {"other_energy": 0.45, "other_melodic": 0.72, "phrase_count": 4, "unique_pitch_classes": 5},
                40.0: {"other_energy": 0.12, "other_melodic": 0.1, "phrase_count": 0, "unique_pitch_classes": 1},
            }[start]
            return {
                "window": [start, start + 20.0],
                "stems": {
                    "other": {"energy_ratio": score_table["other_energy"], "melodic_ratio": score_table["other_melodic"]},
                    "vocals": {"melodic_ratio": 0.1},
                    "piano": {"melodic_ratio": 0.05},
                    "guitar": {"melodic_ratio": 0.05},
                    "bass": {"melodic_ratio": 0.1},
                },
                "note_patterns": {
                    "other": {
                        "phrase_count": score_table["phrase_count"],
                        "unique_pitch_classes": score_table["unique_pitch_classes"],
                    }
                },
            }

        profile_mock.side_effect = fake_profile

        result = pick_top_source_windows("audio/example.mp3", top_k=2)

        self.assertEqual([item["window"][0] for item in result["picked"]], [10.0, 30.0])

    def test_profile_change_score_is_low_for_similar_profiles(self):
        left = _profile(
            energy=[0.5, 0.5],
            melodic_ratio=[0.8, 0.2],
            chroma=[1.0, 0.0, 0.0],
            other_pitch=[1.0, 0.0, 0.0],
            other_interval=[1.0, 0.0],
            other_contour=[0.0, 1.0],
            bass_pitch=[0.0, 1.0],
            bass_interval=[0.2, 0.8],
            other_top=["E"],
            other_chroma=["E"],
            phrase_count=3,
        )
        right = _profile(
            energy=[0.49, 0.51],
            melodic_ratio=[0.79, 0.21],
            chroma=[0.99, 0.01, 0.0],
            other_pitch=[0.99, 0.01, 0.0],
            other_interval=[0.98, 0.02],
            other_contour=[0.01, 0.99],
            bass_pitch=[0.02, 0.98],
            bass_interval=[0.25, 0.75],
            other_top=["E"],
            other_chroma=["E"],
            phrase_count=4,
        )
        distant = _profile(
            energy=[0.0, 1.0],
            melodic_ratio=[0.1, 0.9],
            chroma=[0.0, 1.0, 0.0],
            other_pitch=[0.0, 1.0, 0.0],
            other_interval=[0.0, 1.0],
            other_contour=[1.0, 0.0],
            bass_pitch=[1.0, 0.0],
            bass_interval=[1.0, 0.0],
            other_top=["F#"],
            other_chroma=["F#"],
            phrase_count=0,
        )

        self.assertLess(_profile_change_score(left, right), 0.2)
        self.assertGreater(_profile_change_score(left, distant), 0.5)


if __name__ == "__main__":
    unittest.main()
