from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.render_no_reference_minus_segment_switch_candidate import (
    build_minus_segment_switch_candidate,
    main,
)


SAMPLE_RATE = 8000


def write_wav(path: Path, value: float, *, seconds: float = 4.0) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.full((int(seconds * SAMPLE_RATE), 2), value, dtype=np.float32)
    sf.write(path, audio, SAMPLE_RATE, subtype="FLOAT")
    return str(path)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def candidate_manifest(
    tmp_path: Path,
    candidate_id: str,
    *,
    vocal_value: float,
    minus_value: float,
    blocker_segments: set[int],
    blocker_severity_overrides: dict[int, float] | None = None,
    blocker_issue_overrides: dict[int, list[str]] | None = None,
    safe_segment_overrides: dict[int, float] | None = None,
) -> dict:
    blocker_severity_overrides = blocker_severity_overrides or {}
    blocker_issue_overrides = blocker_issue_overrides or {}
    safe_segment_overrides = safe_segment_overrides or {}
    segments = []
    for index in range(4):
        is_blocker = index in blocker_segments
        severity = (
            blocker_severity_overrides.get(index, 0.2)
            if is_blocker
            else safe_segment_overrides.get(index, 0.0)
        )
        segments.append(
            {
                "segment_id": f"segment_{index:04d}",
                "window_sec": [float(index), float(index + 1)],
                "blocker": is_blocker,
                "issues": blocker_issue_overrides.get(index, ["vocal_projection_proxy"]) if is_blocker else [],
                "severity_score": severity,
                "metrics": {
                    "vocal_projection_db_vs_instrumental": -18.0 if is_blocker else -26.0,
                    "vocal_stem_corr_abs": 0.12 if is_blocker else 0.04,
                },
            }
        )
    return {
        "mode": "instrumental_self_judge_gate",
        "source_audio_path": write_wav(tmp_path / "mix.wav", 0.5),
        "candidate": {
            "label": candidate_id,
            "vocal_path": write_wav(tmp_path / candidate_id / "vocal.wav", vocal_value),
            "instrumental_path": write_wav(tmp_path / candidate_id / "minus.wav", minus_value),
        },
        "overall": {
            "status": "proxy_blocked" if blocker_segments else "proxy_pass",
            "blocker_count": len(blocker_segments),
            "segment_count": 4,
        },
        "delivery_contract": {
            "delivery_allowed": False,
            "presentation_status": "diagnostic_self_judge_not_for_listen_pack",
        },
        "segments": segments,
    }


def test_minus_segment_switch_overlays_only_base_blocker_segments(tmp_path: Path) -> None:
    base = candidate_manifest(tmp_path, "base", vocal_value=0.1, minus_value=0.2, blocker_segments={1})
    alt = candidate_manifest(tmp_path, "alt", vocal_value=0.9, minus_value=0.8, blocker_segments=set())

    result = build_minus_segment_switch_candidate(
        [base, alt],
        output_dir=tmp_path / "out",
        base_candidate_id="base",
        crossfade_ms=0,
    )

    assert result["render_status"] == "rendered_minus_segment_switch_candidate"
    assert result["delivery_allowed"] is False
    assert result["can_publish_as_result"] is False
    assert result["can_mark_release_ready"] is False
    assert result["requires_post_render_minus_self_judge"] is True
    assert result["rendered_vocal_companion_path"]
    assert result["candidate_row"]["vocal_path"] == result["rendered_vocal_companion_path"]
    assert result["candidate_row"]["selected_vocal_path"] == result["rendered_vocal_companion_path"]
    assert result["candidate_row"]["minus_path"] == result["rendered_minus_candidate_path"]

    minus, sample_rate = sf.read(result["rendered_minus_candidate_path"], always_2d=True)
    vocal, vocal_rate = sf.read(result["rendered_vocal_companion_path"], always_2d=True)
    assert sample_rate == SAMPLE_RATE
    assert vocal_rate == SAMPLE_RATE
    assert minus.shape == (SAMPLE_RATE * 4, 2)
    assert vocal.shape == (SAMPLE_RATE * 4, 2)
    assert np.allclose(minus[:SAMPLE_RATE], 0.2, atol=1e-4)
    assert np.allclose(minus[SAMPLE_RATE : SAMPLE_RATE * 2], 0.8, atol=1e-4)
    assert np.allclose(minus[SAMPLE_RATE * 2 :], 0.2, atol=1e-4)
    assert np.allclose(vocal[:SAMPLE_RATE], 0.1, atol=1e-4)
    assert np.allclose(vocal[SAMPLE_RATE : SAMPLE_RATE * 2], 0.9, atol=1e-4)
    assert np.allclose(vocal[SAMPLE_RATE * 2 :], 0.1, atol=1e-4)
    assert result["duration_preserved"] is True
    assert result["sample_rate_preserved"] is True
    assert result["channels_preserved"] is True


def test_minus_segment_switch_blocks_missing_base_candidate(tmp_path: Path) -> None:
    alt = candidate_manifest(tmp_path, "alt", vocal_value=0.9, minus_value=0.8, blocker_segments=set())

    result = build_minus_segment_switch_candidate(
        [alt],
        output_dir=tmp_path / "out",
        base_candidate_id="missing",
        crossfade_ms=0,
    )

    assert result["render_status"] == "blocked_missing_base_candidate"
    assert result["audio_exported"] is False
    assert result["rendered_minus_candidate_path"] == ""


def test_minus_segment_switch_blocks_when_no_safe_alternate_exists(tmp_path: Path) -> None:
    base = candidate_manifest(tmp_path, "base", vocal_value=0.1, minus_value=0.2, blocker_segments={1})
    alt = candidate_manifest(tmp_path, "alt", vocal_value=0.9, minus_value=0.8, blocker_segments={1})

    result = build_minus_segment_switch_candidate(
        [base, alt],
        output_dir=tmp_path / "out",
        base_candidate_id="base",
        crossfade_ms=0,
    )

    assert result["render_status"] == "blocked_no_safe_alternate_segments"
    assert result["selected_segment_count"] == 0
    assert result["blocked_segment_count"] == 1
    assert result["audio_exported"] is False


def test_minus_segment_switch_opt_in_uses_lower_severity_blocked_alternate_as_repair_candidate(
    tmp_path: Path,
) -> None:
    base = candidate_manifest(
        tmp_path,
        "base",
        vocal_value=0.1,
        minus_value=0.2,
        blocker_segments={1},
        blocker_severity_overrides={1: 0.7},
        blocker_issue_overrides={1: ["full_mix_like_risk_proxy"]},
    )
    alt = candidate_manifest(
        tmp_path,
        "alt",
        vocal_value=0.9,
        minus_value=0.8,
        blocker_segments={1},
        blocker_severity_overrides={1: 0.2},
        blocker_issue_overrides={1: ["vocal_projection_proxy"]},
    )

    result = build_minus_segment_switch_candidate(
        [base, alt],
        output_dir=tmp_path / "out",
        base_candidate_id="base",
        crossfade_ms=0,
        allow_repairable_blocked_alternate=True,
    )

    assert result["render_status"] == "rendered_minus_segment_switch_candidate"
    assert result["selected_segment_count"] == 1
    assert result["blocked_segment_count"] == 0
    assert result["repairable_blocked_segment_count"] == 1
    assert result["delivery_allowed"] is False
    assert result["can_publish_as_result"] is False
    selected_segment = result["selected_segments"][0]
    assert selected_segment["selected_candidate_id"] == "alt"
    assert selected_segment["selected_segment_blocker"] is True
    assert selected_segment["selected_requires_post_repair"] is True
    assert selected_segment["selection_reason"] == "alternate_segment_has_lower_self_judge_severity_requires_post_repair"


def test_minus_segment_switch_cli_writes_deterministic_manifest(tmp_path: Path) -> None:
    base_path = write_json(
        tmp_path / "base_manifest.json",
        candidate_manifest(tmp_path, "base", vocal_value=0.1, minus_value=0.2, blocker_segments={1}),
    )
    alt_path = write_json(
        tmp_path / "alt_manifest.json",
        candidate_manifest(tmp_path, "alt", vocal_value=0.9, minus_value=0.8, blocker_segments=set()),
    )
    out_dir = tmp_path / "out"
    manifest_json = out_dir / "manifest.json"
    summary_md = out_dir / "summary.md"
    args = [
        "--self-judge-manifest",
        str(base_path),
        "--self-judge-manifest",
        str(alt_path),
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
    assert payload["render_status"] == "rendered_minus_segment_switch_candidate"
    assert "- render_status: rendered_minus_segment_switch_candidate" in summary_md.read_text(encoding="utf-8")
