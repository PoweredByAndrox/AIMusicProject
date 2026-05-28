import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.segment_rating_remap import build_rating_safe_segment_remap, evaluate_rating_safe_remap_acceptance


class TestSegmentRatingRemap(unittest.TestCase):
    def test_duplicates_old_rating_to_safe_split_children(self):
        old_examples = [
            {
                "example_id": 10,
                "start_time": 0.0,
                "end_time": 40.0,
                "rating": None,
                "is_positive": None,
                "user_ratings": [
                    {"user_id": 14, "rating": 5.0, "is_positive": True},
                ],
            }
        ]
        new_examples = [
            {"example_id": 101, "start_time": 0.0, "end_time": 22.0},
            {"example_id": 102, "start_time": 22.0, "end_time": 40.0},
        ]

        payload = build_rating_safe_segment_remap(old_examples, new_examples)

        self.assertEqual(payload["summary"]["old_examples_remapped"], 1)
        self.assertEqual(payload["summary"]["user_updates_applied"], 2)
        self.assertEqual(payload["summary"]["user_updates_skipped"], 0)
        remapped_ids = sorted(item["pattern_example_id"] for item in payload["user_updates"])
        self.assertEqual(remapped_ids, [101, 102])
        self.assertTrue(all(item["rating"] == 5.0 for item in payload["user_updates"]))
        acceptance = evaluate_rating_safe_remap_acceptance(payload)
        self.assertTrue(acceptance["accepted"])

    def test_skips_conflicting_merge_for_same_user(self):
        old_examples = [
            {
                "example_id": 10,
                "start_time": 0.0,
                "end_time": 20.0,
                "rating": None,
                "is_positive": None,
                "user_ratings": [
                    {"user_id": 14, "rating": 5.0, "is_positive": True},
                ],
            },
            {
                "example_id": 11,
                "start_time": 20.0,
                "end_time": 40.0,
                "rating": None,
                "is_positive": None,
                "user_ratings": [
                    {"user_id": 14, "rating": 1.0, "is_positive": False},
                ],
            },
        ]
        new_examples = [
            {"example_id": 201, "start_time": 0.0, "end_time": 40.0},
        ]

        payload = build_rating_safe_segment_remap(old_examples, new_examples)

        self.assertEqual(payload["summary"]["old_examples_remapped"], 2)
        self.assertEqual(payload["summary"]["user_updates_applied"], 0)
        self.assertEqual(payload["summary"]["user_updates_skipped"], 1)
        skipped_reason = payload["summary"]["user_skipped_debug"][0]["reason"]
        self.assertEqual(skipped_reason, "user_mixed_sentiment")
        acceptance = evaluate_rating_safe_remap_acceptance(payload)
        self.assertFalse(acceptance["accepted"])
        self.assertIn("user_rating_conflicts", acceptance["reasons"])


if __name__ == "__main__":
    unittest.main()
