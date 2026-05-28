import unittest

from app.main import (
    BANQUET_QUERY_MODE_HARMONIC_RESIDUAL_SINGLE,
    select_banquet_query_candidates,
)


def _window(similarity: float, start_sec: float = 0.0) -> dict[str, object]:
    return {
        "start_sec": start_sec,
        "end_sec": start_sec + 8.0,
        "rms": 0.2,
        "similarity": similarity,
    }


def _ranked_windows(scores: dict[str, float]) -> dict[str, list[dict[str, object]]]:
    return {
        component_name: [_window(score), _window(score - 0.05, start_sec=8.0)]
        for component_name, score in scores.items()
    }


class BanquetQuerySelectionTests(unittest.TestCase):
    def test_harmonic_residual_mode_uses_only_harmonic_residual_bus(self) -> None:
        scores = {
            "bus_other": 0.99,
            "bus_guitar": 0.98,
            "bus_piano": 0.97,
            "input_harmonic": 0.96,
            "bus_harmonic_residual": 0.5,
        }
        ranked_windows = _ranked_windows(scores)

        selection = select_banquet_query_candidates(
            "strings_orchestra",
            scores,
            {component_name: windows[0] for component_name, windows in ranked_windows.items()},
            ranked_windows,
            query_mode=BANQUET_QUERY_MODE_HARMONIC_RESIDUAL_SINGLE,
        )

        self.assertEqual(selection["query_mode"], BANQUET_QUERY_MODE_HARMONIC_RESIDUAL_SINGLE)
        self.assertEqual(selection["top_k"], 1)
        self.assertEqual(selection["best"]["component"], "bus_harmonic_residual")
        self.assertEqual([item["component"] for item in selection["ensemble"]], ["bus_harmonic_residual"])
        self.assertEqual([item["component"] for item in selection["ranked"]], ["bus_harmonic_residual", "bus_harmonic_residual"])
        self.assertEqual(
            {item["component"] for item in selection["blocked_components"]},
            {"bus_other", "bus_guitar", "bus_piano", "input_harmonic"},
        )

    def test_default_mode_can_use_prompt_biased_proxy_buses(self) -> None:
        scores = {
            "bus_other": 0.7,
            "bus_guitar": 0.68,
            "bus_piano": 0.66,
            "bus_harmonic_residual": 0.4,
        }
        ranked_windows = _ranked_windows(scores)

        selection = select_banquet_query_candidates(
            "strings_orchestra",
            scores,
            {component_name: windows[0] for component_name, windows in ranked_windows.items()},
            ranked_windows,
        )

        selected_components = [item["component"] for item in selection["ensemble"]]
        self.assertIn("bus_other", selected_components)
        self.assertIn("bus_guitar", selected_components)
        self.assertEqual(selection["query_mode"], "default")
        self.assertEqual(selection["blocked_components"], [])

    def test_harmonic_residual_mode_fails_fast_when_bus_is_missing(self) -> None:
        scores = {
            "bus_other": 0.99,
            "input_harmonic": 0.96,
        }
        ranked_windows = _ranked_windows(scores)

        with self.assertRaisesRegex(RuntimeError, "query_mode=harmonic_residual_single"):
            select_banquet_query_candidates(
                "strings_orchestra",
                scores,
                {component_name: windows[0] for component_name, windows in ranked_windows.items()},
                ranked_windows,
                query_mode=BANQUET_QUERY_MODE_HARMONIC_RESIDUAL_SINGLE,
            )

    def test_unknown_query_mode_fails_fast(self) -> None:
        scores = {"bus_harmonic_residual": 0.5}
        ranked_windows = _ranked_windows(scores)

        with self.assertRaisesRegex(ValueError, "Unsupported Banquet query_mode"):
            select_banquet_query_candidates(
                "strings_orchestra",
                scores,
                {"bus_harmonic_residual": ranked_windows["bus_harmonic_residual"][0]},
                ranked_windows,
                query_mode="unknown_mode",
            )


if __name__ == "__main__":
    unittest.main()
