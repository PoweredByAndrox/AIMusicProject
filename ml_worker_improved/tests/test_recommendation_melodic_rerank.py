import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.recommendation_melodic_rerank import (
    score_section_aware_candidate,
    summarize_family_seed_coverage,
    summarize_top_family_matches,
)


class TestRecommendationMelodicRerank(unittest.TestCase):
    def test_seed_coverage_counts_family_matches(self):
        coverage = summarize_family_seed_coverage(
            [
                {
                    "similarity": 0.61,
                    "seed": {"track_id": 1, "start": 10.0, "end": 20.0, "type": "verse"},
                    "candidate_segment": {"start": 70.0, "end": 90.0, "type": "melodic"},
                },
                {
                    "similarity": 0.42,
                    "seed": {"track_id": 2, "start": 30.0, "end": 40.0, "type": "verse"},
                    "candidate_segment": {"start": 70.0, "end": 90.0, "type": "melodic"},
                },
                {
                    "similarity": 0.73,
                    "seed": {"track_id": 3, "start": 50.0, "end": 65.0, "type": "drop_bass"},
                    "candidate_segment": {"start": 120.0, "end": 140.0, "type": "drop_bass"},
                },
            ]
        )

        self.assertEqual(coverage["hook"]["seed_count"], 2)
        self.assertEqual(coverage["hook"]["matched_seed_count"], 1)
        self.assertAlmostEqual(coverage["hook"]["coverage_ratio"], 0.5, places=4)
        self.assertEqual(coverage["drop"]["seed_count"], 1)
        self.assertEqual(coverage["drop"]["matched_seed_count"], 1)

    def test_drop_family_penalizes_hook_only_candidate(self):
        result = score_section_aware_candidate(
            embedding_score=0.95,
            positive_matches=[
                {
                    "similarity": 0.84,
                    "seed": {"track_id": 9152, "type": "verse"},
                    "candidate_segment": {"start": 77.23, "end": 98.75, "type": "melodic"},
                },
                {
                    "similarity": 0.31,
                    "seed": {"track_id": 4230, "type": "drop_bass"},
                    "candidate_segment": {"start": 140.13, "end": 191.24, "type": "melodic"},
                },
            ],
            negative_matches=[],
        )

        self.assertAlmostEqual(result["section_scores"]["hook"]["margin"], 0.84, places=4)
        self.assertAlmostEqual(result["section_scores"]["drop"]["margin"], 0.0, places=4)
        self.assertAlmostEqual(result["melodic_margin"], 0.504, places=4)

    def test_same_segment_conflict_applies_penalty_and_flag(self):
        result = score_section_aware_candidate(
            embedding_score=0.94,
            positive_matches=[
                {
                    "similarity": 0.86,
                    "seed": {"track_id": 9152, "type": "verse"},
                    "candidate_segment": {"start": 77.23, "end": 98.75, "type": "melodic"},
                }
            ],
            negative_matches=[
                {
                    "similarity": 0.81,
                    "seed": {"track_id": 5159, "type": "melodic"},
                    "candidate_segment": {"start": 77.23, "end": 98.75, "type": "melodic"},
                }
            ],
        )

        self.assertTrue(result["same_segment_conflict"])
        self.assertAlmostEqual(result["same_segment_conflict_penalty"], 0.06, places=4)
        self.assertLess(result["combined_score"], 0.55)

    def test_top_family_matches_keeps_ranked_drop_matches(self):
        ranked = summarize_top_family_matches(
            [
                {
                    "similarity": 0.51,
                    "seed": {"track_id": 1, "type": "drop_bass"},
                    "candidate_segment": {"start": 10.0, "end": 20.0, "type": "drop_bass"},
                },
                {
                    "similarity": 0.73,
                    "seed": {"track_id": 2, "type": "drop_bass"},
                    "candidate_segment": {"start": 20.0, "end": 30.0, "type": "drop_bass"},
                },
                {
                    "similarity": 0.62,
                    "seed": {"track_id": 3, "type": "drop_bass"},
                    "candidate_segment": {"start": 30.0, "end": 40.0, "type": "drop_bass"},
                },
            ],
            top_k=2,
        )

        self.assertEqual([item["seed"]["track_id"] for item in ranked["drop"]], [2, 3])
        self.assertEqual(ranked["hook"], [])

    def test_drop_family_uses_effective_similarity_over_raw_similarity(self):
        result = score_section_aware_candidate(
            embedding_score=0.0,
            positive_matches=[
                {
                    "similarity": 0.41,
                    "effective_similarity": 0.81,
                    "seed": {"track_id": 4230, "type": "drop_bass"},
                    "candidate_segment": {"start": 10.0, "end": 20.0, "type": "drop_bass"},
                }
            ],
            negative_matches=[
                {
                    "similarity": 0.55,
                    "effective_similarity": 0.45,
                    "seed": {"track_id": 6346, "type": "drop_bass"},
                    "candidate_segment": {"start": 10.0, "end": 20.0, "type": "drop_bass"},
                }
            ],
            embedding_weight=0.0,
        )

        self.assertAlmostEqual(result["section_scores"]["drop"]["positive_similarity"], 0.81, places=4)
        self.assertAlmostEqual(result["section_scores"]["drop"]["negative_similarity"], 0.45, places=4)
        self.assertGreater(result["section_scores"]["drop"]["margin"], 0.48)


if __name__ == "__main__":
    unittest.main()
