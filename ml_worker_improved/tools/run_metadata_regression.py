import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from worker.audio_analyzer import AudioAnalyzer


def _norm_text(value) -> str:
    return str(value or "").strip().lower()


def _top_tags(items) -> List[str]:
    out: List[str] = []
    for item in items or []:
        tag = item.get("tag") if isinstance(item, dict) else None
        if tag:
            out.append(_norm_text(tag))
    return out


def main() -> int:
    require_docker_runtime(
        tool_name="run_metadata_regression.py",
        entrypoint_path="/app/tools/run_metadata_regression.py",
    )
    ap = argparse.ArgumentParser(description="Run manual metadata regression for genre/mood/voice predictions.")
    ap.add_argument("spec", help="Path to metadata_regression.json")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        raise SystemExit(f"Spec file not found: {spec_path}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cases = spec.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise SystemExit("Spec has no cases.")

    analyzer = AudioAnalyzer()
    failures: List[str] = []
    annotated_cases = 0
    pending_cases = 0

    for case in cases:
        expected: Dict[str, object] = case.get("expected") or {}
        expected_genre = expected.get("genre")
        expected_mood = expected.get("mood")
        expected_has_vocals = expected.get("has_vocals")

        if expected_genre is None and expected_mood is None and expected_has_vocals is None:
            pending_cases += 1
            continue

        annotated_cases += 1
        audio_path = case["audio_path"]
        result = analyzer.analyze_full(audio_path)

        if expected_genre is not None:
            expected_genre_norm = _norm_text(expected_genre)
            predicted_genre_norm = _norm_text(result.get("genre"))
            genre_tags = _top_tags(result.get("genre_tags"))
            if expected_genre_norm != predicted_genre_norm and expected_genre_norm not in genre_tags:
                failures.append(
                    f"{case['name']}: genre expected={expected_genre!r}, got={result.get('genre')!r}, "
                    f"top_tags={result.get('genre_tags')}"
                )

        if expected_mood is not None:
            expected_mood_norm = _norm_text(expected_mood)
            predicted_mood_norm = _norm_text(result.get("mood"))
            mood_tags = _top_tags(result.get("mood_tags"))
            if expected_mood_norm != predicted_mood_norm and expected_mood_norm not in mood_tags:
                failures.append(
                    f"{case['name']}: mood expected={expected_mood!r}, got={result.get('mood')!r}, "
                    f"top_tags={result.get('mood_tags')}"
                )

        if expected_has_vocals is not None:
            if bool(expected_has_vocals) != bool(result.get("has_vocals")):
                failures.append(
                    f"{case['name']}: has_vocals expected={bool(expected_has_vocals)}, "
                    f"got={bool(result.get('has_vocals'))}"
                )

    print(f"annotated_cases={annotated_cases}")
    print(f"pending_cases={pending_cases}")

    if annotated_cases == 0:
        print("PENDING: metadata_regression.json has no manual labels yet.")
        return 3

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 2

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
