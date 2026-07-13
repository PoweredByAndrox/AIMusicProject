from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.render_no_reference_full_track_segment_switch_candidate import (
    build_full_track_segment_switch_candidate,
    main,
)


SAMPLE_RATE = 8000


def write_wav(path: Path, value: float, *, seconds: float = 4.0) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.full((int(seconds * SAMPLE_RATE), 2), value, dtype=np.float32)
    sf.write(path, audio, SAMPLE_RATE, subtype="FLOAT")
    return str(path)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def candidate_row(tmp_path: Path, candidate_id: str, *, vocal_value: float, minus_value: float, seconds: float = 4.0) -> dict:
    minus_path = write_wav(tmp_path / candidate_id / "minus.wav", minus_value, seconds=seconds)
    return {
        "track_id": "track-a",
        "candidate_id": candidate_id,
        "mix_path": write_wav(tmp_path / "mix.wav", 0.5, seconds=seconds),
        "vocal_path": write_wav(tmp_path / candidate_id / "vocal.wav", vocal_value, seconds=seconds),
        "instrumental_path": minus_path,
        "minus_path": minus_path,
        "source_minus_path": minus_path,
        "base_minus_path": minus_path,
        "is_complete_pair": True,
    }


def window_metric(
    index: int,
    score: float,
    *,
    decision: str | None = None,
    reasons: list[str] | None = None,
    duration_sec: float = 1.0,
) -> dict:
    window = {
        "start_sec": float(index),
        "duration_sec": duration_sec,
        "minus_score": score,
        "vocal_score": score,
        "window_fitness": score,
        "reasons": list(reasons or ["no_reference_proxy_passed"]),
    }
    if decision is not None:
        window["window_decision"] = decision
    return window


def judge_row(candidate_id: str, scores: list[float | dict]) -> dict:
    return {
        "track_id": "track-a",
        "candidate_id": candidate_id,
        "window_metrics": [
            item if isinstance(item, dict) else window_metric(index, float(item))
            for index, item in enumerate(scores)
        ],
    }


def test_full_track_segment_switch_preserves_base_duration_and_overlays_selected_window(tmp_path: Path) -> None:
    base = candidate_row(tmp_path, "base", vocal_value=0.1, minus_value=0.2)
    alt = candidate_row(tmp_path, "alt", vocal_value=0.7, minus_value=0.8)

    result = build_full_track_segment_switch_candidate(
        [base, alt],
        [judge_row("base", [0.8, 0.1, 0.8, 0.8]), judge_row("alt", [0.1, 0.95, 0.1, 0.1])],
        output_dir=tmp_path / "out",
        base_candidate_id="base",
        min_window_score=0.72,
        crossfade_ms=0,
    )

    assert result["render_status"] == "rendered_full_track_candidate"
    assert result["delivery_allowed"] is False
    assert result["can_publish_as_result"] is False
    assert result["can_mark_release_ready"] is False
    assert result["requires_dense_post_reconciliation_judge"] is True
    assert result["candidate_row"]["final_decision"] == "pair_reconciliation_candidate"
    assert result["candidate_row"]["requires_dense_post_reconciliation_judge"] is True
    assert result["base_minus_path"] == base["minus_path"]
    assert result["source_minus_path"] == base["minus_path"]
    assert result["candidate_row"]["base_minus_path"] == base["minus_path"]
    assert result["candidate_row"]["source_minus_path"] == base["minus_path"]
    assert Path(result["rendered_vocal_candidate_path"]).name == "fts_voc.wav"
    assert Path(result["rendered_minus_candidate_path"]).name == "fts_min.wav"
    assert Path(result["candidate_row_jsonl"]).name == "fts_row.jsonl"
    assert len(Path(result["rendered_vocal_candidate_path"]).name) <= 32
    assert len(Path(result["rendered_minus_candidate_path"]).name) <= 32
    assert len(Path(result["candidate_row_jsonl"]).name) <= 32

    vocal, vocal_rate = sf.read(result["rendered_vocal_candidate_path"], always_2d=True)
    minus, minus_rate = sf.read(result["rendered_minus_candidate_path"], always_2d=True)
    assert vocal_rate == minus_rate == SAMPLE_RATE
    assert vocal.shape == minus.shape == (SAMPLE_RATE * 4, 2)
    assert result["duration_preserved"] is True
    assert result["duration_sec"] == 4.0

    assert np.allclose(vocal[:SAMPLE_RATE], 0.1, atol=1e-4)
    assert np.allclose(vocal[SAMPLE_RATE : SAMPLE_RATE * 2], 0.7, atol=1e-4)
    assert np.allclose(vocal[SAMPLE_RATE * 2 :], 0.1, atol=1e-4)
    assert np.allclose(minus[SAMPLE_RATE : SAMPLE_RATE * 2], 0.8, atol=1e-4)


def test_full_track_segment_switch_blocks_missing_base_candidate(tmp_path: Path) -> None:
    alt = candidate_row(tmp_path, "alt", vocal_value=0.7, minus_value=0.8)

    result = build_full_track_segment_switch_candidate(
        [alt],
        [judge_row("alt", [0.95, 0.95])],
        output_dir=tmp_path / "out",
        base_candidate_id="missing",
        min_window_score=0.72,
        crossfade_ms=0,
    )

    assert result["render_status"] == "blocked_missing_base_candidate"
    assert result["audio_exported"] is False
    assert result["rendered_vocal_candidate_path"] == ""
    assert result["rendered_minus_candidate_path"] == ""


def test_full_track_segment_switch_blocks_missing_base_minus_provenance(tmp_path: Path) -> None:
    base = candidate_row(tmp_path, "base", vocal_value=0.1, minus_value=0.2)
    base.pop("source_minus_path", None)
    base.pop("base_minus_path", None)
    alt = candidate_row(tmp_path, "alt", vocal_value=0.7, minus_value=0.8)

    result = build_full_track_segment_switch_candidate(
        [base, alt],
        [judge_row("base", [0.8, 0.1]), judge_row("alt", [0.1, 0.95])],
        output_dir=tmp_path / "out",
        base_candidate_id="base",
        min_window_score=0.72,
        crossfade_ms=0,
    )

    assert result["render_status"] == "blocked_missing_base_minus_provenance"
    assert result["dominant_failure_reason"] == "missing_base_minus_provenance"
    assert result["audio_exported"] is False
    assert result["rendered_vocal_candidate_path"] == ""
    assert result["rendered_minus_candidate_path"] == ""


def test_full_track_segment_switch_prefers_safe_base_over_higher_score_alternate(tmp_path: Path) -> None:
    base = candidate_row(tmp_path, "base", vocal_value=0.1, minus_value=0.2)
    alt = candidate_row(tmp_path, "alt", vocal_value=0.7, minus_value=0.8)

    result = build_full_track_segment_switch_candidate(
        [base, alt],
        [
            judge_row("base", [window_metric(0, 0.81, decision="accept_candidate")]),
            judge_row("alt", [window_metric(0, 0.98, decision="select_alternative_candidate")]),
        ],
        output_dir=tmp_path / "out",
        base_candidate_id="base",
        min_window_score=0.72,
        crossfade_ms=0,
        selection_decision_field="window_decision",
        allow_select_alternative=True,
        prefer_base_when_safe=True,
    )

    assert result["selection_prefers_safe_base"] is True
    assert result["render_status"] == "blocked_no_alternate_window_selected"
    assert result["dominant_failure_reason"] == "no_full_track_overlay_improvement"
    assert result["selected_window_count"] == 1
    assert result["overlay_window_count"] == 0
    assert result["selected_windows"][0]["candidate_id"] == "base"
    assert result["selected_windows"][0]["selected_candidate_id"] == "base"
    assert result["blocked_window_count"] == 0


def test_full_track_segment_switch_selects_alternate_when_base_decision_blocked(tmp_path: Path) -> None:
    base = candidate_row(tmp_path, "base", vocal_value=0.1, minus_value=0.2)
    alt = candidate_row(tmp_path, "alt", vocal_value=0.7, minus_value=0.8)

    result = build_full_track_segment_switch_candidate(
        [base, alt],
        [
            judge_row("base", [window_metric(0, 0.94, decision="reject_candidate")]),
            judge_row("alt", [window_metric(0, 0.82, decision="select_alternative_candidate")]),
        ],
        output_dir=tmp_path / "out",
        base_candidate_id="base",
        min_window_score=0.72,
        crossfade_ms=0,
        selection_decision_field="window_decision",
        allow_select_alternative=True,
        prefer_base_when_safe=True,
    )

    assert result["selection_prefers_safe_base"] is True
    assert result["render_status"] == "rendered_full_track_candidate"
    assert result["selected_window_count"] == 1
    assert result["overlay_window_count"] == 1
    assert result["selected_windows"][0]["candidate_id"] == "alt"
    assert result["selected_windows"][0]["selected_candidate_id"] == "alt"
    assert result["blocked_window_count"] == 0


def test_full_track_segment_switch_default_does_not_prefer_base_when_flag_absent(tmp_path: Path) -> None:
    base = candidate_row(tmp_path, "base", vocal_value=0.1, minus_value=0.2)
    alt = candidate_row(tmp_path, "alt", vocal_value=0.7, minus_value=0.8)

    result = build_full_track_segment_switch_candidate(
        [base, alt],
        [
            judge_row("base", [window_metric(0, 0.81, decision="accept_candidate")]),
            judge_row("alt", [window_metric(0, 0.98, decision="select_alternative_candidate")]),
        ],
        output_dir=tmp_path / "out",
        base_candidate_id="base",
        min_window_score=0.72,
        crossfade_ms=0,
        selection_decision_field="window_decision",
        allow_select_alternative=True,
    )

    assert result["selection_prefers_safe_base"] is False
    assert result["render_status"] == "rendered_full_track_candidate"
    assert result["selected_window_count"] == 1
    assert result["overlay_window_count"] == 1
    assert result["selected_windows"][0]["candidate_id"] == "alt"
    assert result["selected_windows"][0]["selected_candidate_id"] == "alt"


def test_full_track_segment_switch_cli_writes_deterministic_manifest(tmp_path: Path) -> None:
    base = candidate_row(tmp_path, "base", vocal_value=0.1, minus_value=0.2)
    alt = candidate_row(tmp_path, "alt", vocal_value=0.7, minus_value=0.8)
    candidates_jsonl = write_jsonl(tmp_path / "candidates.jsonl", [alt, base])
    judge_jsonl = write_jsonl(tmp_path / "judge.jsonl", [judge_row("base", [0.8, 0.1]), judge_row("alt", [0.1, 0.95])])
    out_dir = tmp_path / "out"
    manifest_json = out_dir / "manifest.json"
    summary_md = out_dir / "summary.md"
    args = [
        "--candidate-pool-jsonl",
        str(candidates_jsonl),
        "--windowed-judge-jsonl",
        str(judge_jsonl),
        "--output-dir",
        str(out_dir),
        "--manifest-json",
        str(manifest_json),
        "--summary-md",
        str(summary_md),
        "--base-candidate-id",
        "base",
        "--crossfade-ms",
        "0",
    ]

    assert main(args) == 0
    first_manifest = manifest_json.read_text(encoding="utf-8")
    assert main(args) == 0
    assert manifest_json.read_text(encoding="utf-8") == first_manifest
    payload = json.loads(first_manifest)
    assert payload["render_status"] == "rendered_full_track_candidate"
    assert payload["base_minus_path"] == base["minus_path"]
    assert payload["source_minus_path"] == base["minus_path"]
    assert Path(payload["rendered_vocal_candidate_path"]).name == "fts_voc.wav"
    assert Path(payload["rendered_minus_candidate_path"]).name == "fts_min.wav"
    assert Path(payload["candidate_row_jsonl"]).name == "fts_row.jsonl"
    assert payload["selection_prefers_safe_base"] is False
    assert "- render_status: rendered_full_track_candidate" in summary_md.read_text(encoding="utf-8")


def test_full_track_segment_switch_cli_sets_prefer_base_manifest_flag(tmp_path: Path) -> None:
    base = candidate_row(tmp_path, "base", vocal_value=0.1, minus_value=0.2)
    alt = candidate_row(tmp_path, "alt", vocal_value=0.7, minus_value=0.8)
    candidates_jsonl = write_jsonl(tmp_path / "candidates.jsonl", [alt, base])
    judge_jsonl = write_jsonl(
        tmp_path / "judge.jsonl",
        [
            judge_row("base", [window_metric(0, 0.81, decision="accept_candidate")]),
            judge_row("alt", [window_metric(0, 0.98, decision="select_alternative_candidate")]),
        ],
    )
    out_dir = tmp_path / "out"
    manifest_json = out_dir / "manifest.json"
    summary_md = out_dir / "summary.md"
    args = [
        "--candidate-pool-jsonl",
        str(candidates_jsonl),
        "--windowed-judge-jsonl",
        str(judge_jsonl),
        "--output-dir",
        str(out_dir),
        "--manifest-json",
        str(manifest_json),
        "--summary-md",
        str(summary_md),
        "--base-candidate-id",
        "base",
        "--crossfade-ms",
        "0",
        "--selection-decision-field",
        "window_decision",
        "--allow-select-alternative",
        "--prefer-base-when-safe",
    ]

    assert main(args) == 0
    payload = json.loads(manifest_json.read_text(encoding="utf-8"))

    assert payload["selection_prefers_safe_base"] is True
    assert payload["render_status"] == "blocked_no_alternate_window_selected"
    assert payload["selected_window_count"] == 1
    assert payload["selected_windows"][0]["candidate_id"] == "base"
