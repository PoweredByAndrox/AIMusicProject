import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.recommendation_lab_gate import (
    evaluate_holdout_gate,
    evaluate_recommendation_lab_gate,
)


class TestRecommendationLabGate(unittest.TestCase):
    def test_gate_fails_when_any_holdout_is_below_threshold(self):
        results = [
            {"track_id": 5037, "title": "Higher", "estimated_like_pct": 49.6, "combined_score": 0.4979},
            {"track_id": 14047, "title": "Home Alone", "estimated_like_pct": 36.3, "combined_score": 0.4295},
            {"track_id": 7111, "title": "Fall Down", "estimated_like_pct": 71.4, "combined_score": 0.6144},
        ]

        summary = evaluate_holdout_gate(results, holdout_track_ids=[5037, 14047], min_like_pct=55.0)

        self.assertFalse(summary["gate_passed"])
        self.assertEqual(len(summary["failing_holdouts"]), 2)
        self.assertEqual(summary["missing_holdouts"], [])

    def test_gate_passes_when_all_holdouts_clear_threshold(self):
        results = [
            {"track_id": 5037, "title": "Higher", "estimated_like_pct": 61.0, "combined_score": 0.61},
            {"track_id": 14047, "title": "Home Alone", "estimated_like_pct": 58.5, "combined_score": 0.58},
        ]

        summary = evaluate_holdout_gate(results, holdout_track_ids=[5037, 14047], min_like_pct=55.0)

        self.assertTrue(summary["gate_passed"])
        self.assertEqual(len(summary["passing_holdouts"]), 2)
        self.assertEqual(summary["failing_holdouts"], [])
        self.assertEqual(summary["missing_holdouts"], [])

    def test_gate_fails_when_top_rank_candidate_has_same_segment_conflict(self):
        results = [
            {
                "track_id": 7173,
                "title": "Under Control",
                "estimated_like_pct": 62.5,
                "combined_score": 0.5644,
                "same_segment_conflict": False,
            },
            {
                "track_id": 7111,
                "title": "Fall Down",
                "estimated_like_pct": 61.6,
                "combined_score": 0.5595,
                "same_segment_conflict": True,
                "best_positive_segment": {"start": 77.23, "end": 98.75, "type": "melodic"},
                "best_negative_segment": {"start": 77.23, "end": 98.75, "type": "melodic"},
            },
            {
                "track_id": 5037,
                "title": "Higher",
                "estimated_like_pct": 55.1,
                "combined_score": 0.5261,
                "same_segment_conflict": False,
            },
            {
                "track_id": 14047,
                "title": "Home Alone",
                "estimated_like_pct": 58.0,
                "combined_score": 0.5300,
                "same_segment_conflict": False,
            },
        ]

        summary = evaluate_recommendation_lab_gate(
            results,
            holdout_track_ids=[5037, 14047],
            min_like_pct=55.0,
            conflict_top_rank_limit=3,
        )

        self.assertFalse(summary["gate_passed"])
        self.assertTrue(summary["holdout_gate"]["gate_passed"])
        self.assertFalse(summary["same_segment_conflict_gate"]["gate_passed"])
        self.assertEqual(summary["same_segment_conflict_gate"]["failing_candidates"][0]["track_id"], 7111)


if __name__ == "__main__":
    unittest.main()
