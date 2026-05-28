import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.recommendation_lab_section_aware import (
    _candidate_segments_for_track,
    _extract_raw_drop_match_features,
    _query_seed_rows,
    _select_discriminative_seed_rows,
)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = list(rows)
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params):
        self.executed.append((query, params))

    def fetchall(self):
        return list(self._rows)


class _FakeConnection:
    def __init__(self, rows):
        self._cursor = _FakeCursor(rows)

    def cursor(self):
        return self._cursor


class _FakeWorker:
    def __init__(self, rows):
        self.db_conn = _FakeConnection(rows)


class TestRecommendationLabSectionAware(unittest.TestCase):
    def test_query_seed_rows_respects_exclude_track_ids(self):
        worker = _FakeWorker(
            [
                (14047, "Home Alone", "Artist", "audio/a.mp3", 10.0, 30.0, "type=verse", 5, None),
                (5037, "Higher", "Artist", "audio/b.mp3", 40.0, 60.0, "type=drop_bass", 5, None),
                (4230, "Ghosts", "Artist", "audio/c.mp3", 70.0, 92.0, "type=drop_bass", 5, None),
            ]
        )

        rows = _query_seed_rows(
            worker,
            user_id=14,
            min_rating=4,
            max_rating=None,
            family="drop",
            limit=3,
            distinct_track_limit=3,
            exclude_track_ids={14047, 5037},
        )

        self.assertEqual([row["track_id"] for row in rows], [4230])

    def test_candidate_segments_keep_short_drop_windows(self):
        worker = _FakeWorker(
            [
                (0.0, 32.0, "type=verse"),
                (40.0, 48.1, "type=drop_bass"),
                (52.0, 63.3, "type=drop_bass"),
                (70.0, 125.0, "type=drop_bass"),
                (130.0, 150.0, "type=ambient"),
            ]
        )

        segments = _candidate_segments_for_track(worker, track_id=14047)

        self.assertEqual([segment["type"] for segment in segments[:3]], ["verse", "drop_bass", "drop_bass"])
        self.assertAlmostEqual(segments[1]["duration"], 8.1, places=1)
        self.assertAlmostEqual(segments[2]["duration"], 11.3, places=1)

    def test_query_seed_rows_enforces_distinct_tracks(self):
        worker = _FakeWorker(
            [
                (4230, "Ghosts A", "Artist", "audio/a.mp3", 10.0, 30.0, "type=drop_bass", 5, None),
                (4230, "Ghosts B", "Artist", "audio/a.mp3", 40.0, 60.0, "type=drop_bass", 5, None),
                (6793, "Set Fire", "Artist", "audio/b.mp3", 70.0, 92.0, "type=drop_bass", 5, None),
            ]
        )

        rows = _query_seed_rows(
            worker,
            user_id=14,
            min_rating=4,
            max_rating=None,
            family="drop",
            limit=3,
            distinct_track_limit=3,
        )

        self.assertEqual([row["track_id"] for row in rows], [4230, 6793])

    def test_discriminative_drop_selection_prefers_lower_opposite_pressure(self):
        positive = [
            {
                "track_id": 100,
                "start": 10.0,
                "end": 30.0,
                "type": "drop_bass",
                "_segment_embedding": np.array([1.0, 0.0], dtype=np.float32),
            },
            {
                "track_id": 200,
                "start": 40.0,
                "end": 60.0,
                "type": "drop_bass",
                "_segment_embedding": np.array([0.0, 1.0], dtype=np.float32),
            },
            {
                "track_id": 300,
                "start": 70.0,
                "end": 90.0,
                "type": "drop_bass",
                "_segment_embedding": np.array([0.0, 0.9], dtype=np.float32),
            },
        ]
        negative = [
            {
                "track_id": 400,
                "start": 15.0,
                "end": 35.0,
                "type": "drop_bass",
                "_segment_embedding": np.array([0.98, 0.02], dtype=np.float32),
            },
            {
                "track_id": 500,
                "start": 45.0,
                "end": 65.0,
                "type": "drop_bass",
                "_segment_embedding": np.array([0.95, 0.05], dtype=np.float32),
            },
        ]

        selected = _select_discriminative_seed_rows(
            positive,
            opposite_seeds=negative,
            limit=2,
            distinct_track_limit=2,
        )

        self.assertEqual({seed["track_id"] for seed in selected}, {200, 300})
        self.assertGreater(selected[0]["discriminative_score"], selected[1]["discriminative_score"])

    def test_discriminative_drop_selection_enforces_distinct_tracks(self):
        positive = [
            {
                "track_id": 100,
                "start": 10.0,
                "end": 30.0,
                "type": "drop_bass",
                "_segment_embedding": np.array([0.0, 1.0], dtype=np.float32),
            },
            {
                "track_id": 100,
                "start": 40.0,
                "end": 60.0,
                "type": "drop_bass",
                "_segment_embedding": np.array([0.0, 0.99], dtype=np.float32),
            },
            {
                "track_id": 200,
                "start": 70.0,
                "end": 90.0,
                "type": "drop_bass",
                "_segment_embedding": np.array([0.6, 0.8], dtype=np.float32),
            },
        ]
        negative = [
            {
                "track_id": 400,
                "start": 15.0,
                "end": 35.0,
                "type": "drop_bass",
                "_segment_embedding": np.array([1.0, 0.0], dtype=np.float32),
            }
        ]

        selected = _select_discriminative_seed_rows(
            positive,
            opposite_seeds=negative,
            limit=3,
            distinct_track_limit=3,
        )

        self.assertEqual([seed["track_id"] for seed in selected], [100, 200])

    def test_raw_drop_match_features_use_stem_audio_similarity(self):
        sample_count = 22050 * 2
        timeline = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
        stems_by_track = {
            "seed": {
                "bass": np.sin(2.0 * np.pi * 55.0 * timeline).astype(np.float32),
                "drums": (np.sign(np.sin(2.0 * np.pi * 4.0 * timeline)) * 0.5).astype(np.float32),
                "other": np.sin(2.0 * np.pi * 220.0 * timeline).astype(np.float32),
            },
            "same": {
                "bass": np.sin(2.0 * np.pi * 55.0 * timeline).astype(np.float32),
                "drums": (np.sign(np.sin(2.0 * np.pi * 4.0 * timeline)) * 0.5).astype(np.float32),
                "other": np.sin(2.0 * np.pi * 220.0 * timeline).astype(np.float32),
            },
            "different": {
                "bass": np.sin(2.0 * np.pi * 90.0 * timeline).astype(np.float32),
                "drums": np.zeros(sample_count, dtype=np.float32),
                "other": np.random.default_rng(0).normal(0.0, 0.2, sample_count).astype(np.float32),
            },
        }

        with patch(
            "worker.recommendation_lab_section_aware._get_cached_track_stems",
            side_effect=lambda audio_path, device="cuda": stems_by_track[str(audio_path)],
        ):
            same = _extract_raw_drop_match_features(
                seed_audio_path="seed",
                seed_start=0.0,
                seed_end=1.0,
                candidate_audio_path="same",
                candidate_start=0.0,
                candidate_end=1.0,
            )
            different = _extract_raw_drop_match_features(
                seed_audio_path="seed",
                seed_start=0.0,
                seed_end=1.0,
                candidate_audio_path="different",
                candidate_start=0.0,
                candidate_end=1.0,
            )

        self.assertGreater(same["raw_drop_similarity"], different["raw_drop_similarity"])
        self.assertGreater(same["raw_drop_similarity"], 0.95)


if __name__ == "__main__":
    unittest.main()
