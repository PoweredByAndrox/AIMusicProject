from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "audit_minus_content_words.py"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
SPEC = importlib.util.spec_from_file_location("audit_minus_content_words", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def transcript(text: str, *, start: float = 0.0, end: float = 2.0, reliable: bool = True):
    tokens = MODULE.normalize_tokens(text)
    return {
        "reliable_token_count": len(tokens) if reliable else 0,
        "segments": [
            {
                "start": start,
                "end": end,
                "text": text,
                "tokens": tokens,
                "reliable": reliable,
            }
        ],
    }


class TestMinusContentWords(unittest.TestCase):
    def test_normalize_tokens_handles_unicode_and_ignores_underscores(self):
        self.assertEqual(
            MODULE.normalize_tokens("Hello, HELLO_мир! 2008"),
            ["hello", "hello", "мир", "2008"],
        )

    def test_pair_confirms_time_aligned_source_matching_words(self):
        result = MODULE.evaluate_transcript_pair(
            transcript("we should be together sharing this love"),
            transcript("should be together"),
            min_matched_tokens=2,
            min_candidate_match_ratio=0.5,
            time_tolerance_sec=1.0,
        )
        self.assertTrue(result["confirmed"])
        self.assertEqual(result["status"], "confirmed_source_matching_words")
        self.assertEqual(result["confirmed_windows"], [[0.0, 2.0]])

    def test_pair_does_not_confirm_unmatched_music_hallucination(self):
        result = MODULE.evaluate_transcript_pair(
            transcript("we should be together sharing this love"),
            transcript("instrumental music applause"),
            min_matched_tokens=2,
            min_candidate_match_ratio=0.5,
            time_tolerance_sec=1.0,
        )
        self.assertFalse(result["confirmed"])
        self.assertEqual(result["status"], "inconclusive_unmatched_candidate_words")

    def test_no_candidate_words_never_proves_clean_content(self):
        pair = MODULE.evaluate_transcript_pair(
            transcript("we should be together"),
            transcript("", reliable=False),
            min_matched_tokens=2,
            min_candidate_match_ratio=0.5,
            time_tolerance_sec=1.0,
        )
        aggregate = MODULE.aggregate_candidate(
            [
                {"model_id": "base", "runtime_status": "completed", "comparison": pair},
                {"model_id": "small", "runtime_status": "completed", "comparison": pair},
            ],
            min_model_confirmations=2,
        )
        self.assertEqual(aggregate["status"], "content_words_not_detected")
        self.assertFalse(aggregate["clean_content_proven"])
        self.assertFalse(aggregate["repair_eligible"])

    def test_two_model_confirmation_is_required_for_repair(self):
        confirmed = {"confirmed": True, "confirmed_windows": [[0.0, 2.0]]}
        absent = {"confirmed": False, "status": "words_not_detected"}
        one_vote = MODULE.aggregate_candidate(
            [
                {"model_id": "base", "runtime_status": "completed", "comparison": confirmed},
                {"model_id": "small", "runtime_status": "completed", "comparison": absent},
            ],
            min_model_confirmations=2,
        )
        self.assertEqual(one_vote["status"], "content_inconclusive_single_model_evidence")
        self.assertFalse(one_vote["repair_eligible"])

        two_votes = MODULE.aggregate_candidate(
            [
                {"model_id": "base", "runtime_status": "completed", "comparison": confirmed},
                {"model_id": "small", "runtime_status": "completed", "comparison": confirmed},
            ],
            min_model_confirmations=2,
        )
        self.assertEqual(two_votes["status"], "content_leak_confirmed")
        self.assertTrue(two_votes["repair_eligible"])
        self.assertFalse(two_votes["clean_content_proven"])

    def test_runtime_failure_is_fail_closed(self):
        aggregate = MODULE.aggregate_candidate(
            [
                {
                    "model_id": "base",
                    "runtime_status": "failed",
                    "error": "model load failed",
                },
                {
                    "model_id": "small",
                    "runtime_status": "completed",
                    "comparison": {"confirmed": False, "status": "words_not_detected"},
                },
            ],
            min_model_confirmations=2,
        )
        self.assertEqual(aggregate["status"], "content_unknown_runtime_failure")
        self.assertFalse(aggregate["clean_content_proven"])

    def test_materialized_segment_reliability_uses_probability_guards(self):
        segment = SimpleNamespace(
            start=0.0,
            end=1.5,
            text="source matching words",
            avg_logprob=-0.5,
            no_speech_prob=0.1,
            compression_ratio=1.1,
            words=[],
        )
        info = SimpleNamespace(language="en", language_probability=0.99, duration=1.5)
        result = MODULE.materialize_transcript(
            [segment],
            info,
            max_no_speech_prob=0.65,
            min_avg_logprob=-1.2,
            max_compression_ratio=2.4,
        )
        self.assertEqual(result["reliable_token_count"], 3)
        self.assertTrue(result["segments"][0]["reliable"])

    def test_output_path_must_be_in_approved_artifact_root(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "source.flac"
            candidate = root / "candidate.flac"
            source.write_bytes(b"source")
            candidate.write_bytes(b"candidate")
            with self.assertRaisesRegex(ValueError, "approved project artifact root"):
                MODULE.audit_minus_content_words(
                    source_audio_path=source,
                    candidates=[("base", candidate)],
                    output_dir=root / "outside",
                    model_ids=("base",),
                    model_cache_dir=root / "models",
                    runtime_project_root=root,
                    windows_project_root="D:\\AIMusicProject",
                    min_model_confirmations=1,
                    model_loader=lambda *_: object(),
                    transcriber=lambda *_: ([], SimpleNamespace()),
                )


if __name__ == "__main__":
    unittest.main()
