import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.instrument_melody_profile import build_track_instrument_melody_profile


FIXTURE_PATH = Path(__file__).resolve().parent / 'fixtures' / 'track_14056_instrument_melody_profile.json'


class TestTrack14056InstrumentMelodyProfile(unittest.TestCase):
    def test_track_14056_matches_stem_level_melody_baseline(self):
        fixture = json.loads(FIXTURE_PATH.read_text(encoding='utf-8'))
        audio_path = Path(fixture['audio_path'])
        if not audio_path.exists():
            raise unittest.SkipTest(f"Track fixture audio missing: {audio_path}")

        segments = [
            {
                'segment_id': int(segment['segment_id']),
                'start': float(segment['start']),
                'end': float(segment['end']),
                'type': str(segment['type']),
            }
            for segment in fixture['segments']
        ]

        actual = build_track_instrument_melody_profile(audio_path, segments)

        self.assertEqual(len(actual), len(fixture['segments']))

        for actual_segment, expected_segment in zip(actual, fixture['segments']):
            self.assertEqual(actual_segment['segment_id'], int(expected_segment['segment_id']))
            self.assertEqual(actual_segment['type'], expected_segment['type'])

            for stem_name, expectation in expected_segment['assertions'].items():
                actual_stem = actual_segment['stems'][stem_name]
                min_energy, max_energy = expectation['energy_ratio_range']
                self.assertTrue(
                    min_energy <= actual_stem['energy_ratio'] <= max_energy,
                    f"segment={actual_segment['segment_id']} stem={stem_name} "
                    f"energy_ratio={actual_stem['energy_ratio']} outside [{min_energy}, {max_energy}]"
                )

                expected_prefix = expectation.get('top_chroma_prefix')
                if expected_prefix:
                    self.assertEqual(
                        actual_stem['top_chroma_notes'][:len(expected_prefix)],
                        expected_prefix,
                        f"segment={actual_segment['segment_id']} stem={stem_name} "
                        f"top_chroma_notes={actual_stem['top_chroma_notes']}"
                    )

                expected_contains = expectation.get('top_chroma_contains')
                if expected_contains:
                    for note_name in expected_contains:
                        self.assertIn(
                            note_name,
                            actual_stem['top_chroma_notes'][:4],
                            f"segment={actual_segment['segment_id']} stem={stem_name} "
                            f"top_chroma_notes={actual_stem['top_chroma_notes']}"
                        )

                melodic_ratio_min = expectation.get('melodic_ratio_min')
                if melodic_ratio_min is not None:
                    self.assertGreaterEqual(
                        actual_stem['melodic_ratio'],
                        melodic_ratio_min,
                        f"segment={actual_segment['segment_id']} stem={stem_name} "
                        f"melodic_ratio={actual_stem['melodic_ratio']} < {melodic_ratio_min}"
                    )

                melodic_ratio_max = expectation.get('melodic_ratio_max')
                if melodic_ratio_max is not None:
                    self.assertLessEqual(
                        actual_stem['melodic_ratio'],
                        melodic_ratio_max,
                        f"segment={actual_segment['segment_id']} stem={stem_name} "
                        f"melodic_ratio={actual_stem['melodic_ratio']} > {melodic_ratio_max}"
                    )
