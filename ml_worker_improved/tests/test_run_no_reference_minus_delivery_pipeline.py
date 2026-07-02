from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.artifact_status_contract import can_promote_to_listen_result
from tools.run_no_reference_minus_delivery_pipeline import (
    choose_base_candidate_id_from_score_matrix,
    main,
    run_minus_delivery_pipeline,
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


def assert_pipeline_requires_aggregate_gate(manifest: dict) -> None:
    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "aggregate_machine_gate_required_not_for_listen_pack"
    assert manifest["release_status"] == "not_release_ready"
    assert manifest["dominant_failure_reason"] == "requires_staged_aggregate_validation_and_machine_gate"
    assert manifest["listen_source_promotable"] is True
    assert manifest["source_listen_preview_delivery_status"] == "accepted"
    assert manifest["source_listen_preview_delivery_allowed"] is True
    assert manifest["source_listen_preview_presentation_status"] == "accepted_for_listen_pack"
    assert manifest["source_listen_preview_release_status"] == "listen_preview_only"
    assert manifest["requires_staged_aggregate_machine_gate"] is True
    assert manifest["final_delivery_blockers"] == ["requires_staged_aggregate_validation_and_machine_gate"]


def self_judge_manifest(
    tmp_path: Path,
    candidate_id: str,
    *,
    minus_value: float,
    blocker_segments: set[int],
) -> dict:
    segments = []
    for index in range(4):
        blocker = index in blocker_segments
        segments.append(
            {
                "segment_id": f"segment_{index:04d}",
                "window_sec": [float(index), float(index + 1)],
                "blocker": blocker,
                "issues": ["vocal_projection_proxy"] if blocker else [],
                "severity_score": 0.2 if blocker else 0.0,
            }
        )
    return {
        "mode": "instrumental_self_judge_gate",
        "source_audio_path": write_wav(tmp_path / "mix.wav", 0.5),
        "candidate": {
            "label": candidate_id,
            "vocal_path": write_wav(tmp_path / candidate_id / "vocal.wav", 0.1),
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


def patch_accepting_gates(monkeypatch) -> None:
    def fake_judge_instrumental(**kwargs):
        candidate_path = str(kwargs["candidate_instrumental_path"])
        vocal_path = str(kwargs["candidate_vocal_path"])
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "mode": "instrumental_self_judge_gate",
            "source_audio_path": str(kwargs["source_audio_path"]),
            "candidate": {
                "label": str(kwargs["candidate_label"]),
                "instrumental_path": candidate_path,
                "vocal_path": vocal_path,
            },
            "overall": {"status": "proxy_pass", "blocker_count": 0, "segment_count": 4},
            "delivery_contract": {"delivery_allowed": False},
        }
        write_json(output_dir / "instrumental_self_judge_manifest.json", manifest)
        return manifest

    def fake_judge_candidate_rows(rows, *, score_mode):
        assert score_mode == "minus"
        row = rows[0]
        detail_gate = {
            "schema_version": 1,
            "mode": "minus_musical_detail_preservation_gate",
            "status": "proxy_pass",
            "evidence_type": "no_reference_minus_detail_proxy",
            "required_metrics": [
                "instrumental_integrity",
                "transient_preservation",
                "spectral_balance",
                "phase_stereo_stability",
                "mixture_consistency_score",
            ],
            "metrics": {
                "instrumental_integrity": 0.92,
                "transient_preservation": 0.9,
                "spectral_balance": 0.88,
                "phase_stereo_stability": 0.86,
                "mixture_consistency_score": 0.91,
            },
            "blockers": [],
            "can_mark_release_ready": False,
            "can_publish_as_release": False,
        }
        return {
            "rows": [
                {
                    **row,
                    "score_mode": "minus",
                    "decision": "accept_candidate",
                    "minus_decision": "accept_candidate",
                    "minus_score": 0.91,
                    "minus_blockers": [],
                    "window_count": 12,
                    "overall_no_reference_score": 0.91,
                    "confidence": 0.9,
                    "reasons": ["no_reference_proxy_passed"],
                    "minus_musical_detail_preservation_gate": detail_gate,
                }
            ],
            "summary": {
                "total_tracks": 1,
                "candidates_judged": 1,
                "accepted_candidates": 1,
                "blocked_hard_case": 0,
                "usable_without_stems": 1,
                "decision_counts": {"accept_candidate": 1},
                "top_failure_reasons": {"no_reference_proxy_passed": 1},
                "safety": {},
            },
        }

    def fake_select_candidates(rows, *, score_mode):
        assert score_mode == "minus"
        row = rows[0]
        return {
            "rows": [
                {
                    "track_id": row["track_id"],
                    "score_mode": "minus",
                    "decision": "accept_candidate",
                    "selected_candidate": row,
                    "confidence": 0.9,
                    "reasons": ["selected_by_no_reference_score"],
                    "candidate_count": 1,
                    "accepted_candidate_count": 1,
                    "ranked_candidates": [{"candidate_id": row["candidate_id"], "score_mode": "minus", "selection_score": 0.91}],
                }
            ],
            "summary": {
                "score_mode": "minus",
                "total_tracks": 1,
                "candidates_judged": 1,
                "accepted_candidates": 1,
                "selected_alternatives": 0,
                "segment_blend_needed": 0,
                "blocked_hard_case": 0,
                "usable_without_stems": 1,
                "decision_counts": {"accept_candidate": 1},
                "top_failure_reasons": {"selected_by_no_reference_score": 1},
                "safety": {},
            },
        }

    monkeypatch.setattr("tools.run_no_reference_minus_delivery_pipeline.judge_instrumental", fake_judge_instrumental)
    monkeypatch.setattr("tools.run_no_reference_minus_delivery_pipeline.judge_candidate_rows", fake_judge_candidate_rows)
    monkeypatch.setattr("tools.run_no_reference_minus_delivery_pipeline.select_candidates", fake_select_candidates)


def test_minus_delivery_pipeline_promotes_only_after_existing_gates(tmp_path: Path, monkeypatch) -> None:
    base_path = write_json(
        tmp_path / "base_self_judge.json",
        self_judge_manifest(tmp_path, "base", minus_value=0.2, blocker_segments={1}),
    )
    alt_path = write_json(
        tmp_path / "alt_self_judge.json",
        self_judge_manifest(tmp_path, "alt", minus_value=0.8, blocker_segments=set()),
    )

    patch_accepting_gates(monkeypatch)

    manifest = run_minus_delivery_pipeline(
        self_judge_manifest_paths=[base_path, alt_path],
        output_dir=tmp_path / "out",
        base_candidate_id="base",
        crossfade_ms=0,
        segment_sec=1.0,
    )

    listen_manifest = Path(manifest["final_listen_preview_manifest"])
    assert_pipeline_requires_aggregate_gate(manifest)
    assert manifest["stem_type"] == "minus"
    assert manifest["accepted_count"] == 1
    assert can_promote_to_listen_result(listen_manifest) is True
    assert manifest["safety"]["reference_required"] is False
    assert manifest["safety"]["human_csv_required"] is False
    assert manifest["safety"]["training_started"] is False
    assert manifest["safety"]["fine_tune_started"] is False
    assert manifest["safety"]["models_changed"] is False
    assert manifest["safety"]["cleanup_routes_added"] is False
    assert manifest["safety"]["thresholds_relaxed"] is False


def test_minus_delivery_pipeline_logs_mode_specific_progress(tmp_path: Path, monkeypatch, capsys) -> None:
    base_path = write_json(
        tmp_path / "base_self_judge.json",
        self_judge_manifest(tmp_path, "base", minus_value=0.2, blocker_segments={1}),
    )
    alt_path = write_json(
        tmp_path / "alt_self_judge.json",
        self_judge_manifest(tmp_path, "alt", minus_value=0.8, blocker_segments=set()),
    )

    patch_accepting_gates(monkeypatch)

    run_minus_delivery_pipeline(
        self_judge_manifest_paths=[base_path, alt_path],
        output_dir=tmp_path / "out",
        base_candidate_id="base",
        crossfade_ms=0,
        segment_sec=1.0,
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert any(
        event.get("phase") == "result_judge"
        and event.get("event") == "start"
        and event.get("score_mode") == "minus"
        and event.get("gpu_inference_skipped") is True
        for event in events
    )
    assert any(
        event.get("phase") == "selector"
        and event.get("event") == "start"
        and event.get("score_mode") == "minus"
        and event.get("gpu_inference_skipped") is True
        for event in events
    )
    assert any(
        event.get("phase") == "minus_listen_preview_manifest"
        and event.get("event") == "end"
        and event.get("listen_source_promotable") is True
        for event in events
    )


def test_minus_delivery_pipeline_auto_discovers_manifests_and_base_candidate(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "self_judge_root"
    write_json(
        root / "base" / "instrumental_self_judge_manifest.json",
        self_judge_manifest(tmp_path, "base", minus_value=0.2, blocker_segments={1}),
    )
    write_json(
        root / "noisier" / "instrumental_self_judge_manifest.json",
        self_judge_manifest(tmp_path, "noisier", minus_value=0.3, blocker_segments={1, 2}),
    )
    write_json(
        root / "alt" / "instrumental_self_judge_manifest.json",
        self_judge_manifest(tmp_path, "alt", minus_value=0.8, blocker_segments=set()),
    )
    patch_accepting_gates(monkeypatch)

    assert (
        main(
            [
                "--self-judge-root",
                str(root),
                "--output-dir",
                str(tmp_path / "out"),
                "--crossfade-ms",
                "0",
                "--segment-sec",
                "1",
            ]
        )
        == 1
    )

    final_manifest = json.loads(
        (tmp_path / "out" / "no_reference_minus_delivery_pipeline_manifest.json").read_text(encoding="utf-8")
    )
    assert final_manifest["base_candidate_id"] == "base"
    assert_pipeline_requires_aggregate_gate(final_manifest)


def test_minus_delivery_pipeline_uses_score_matrix_for_base_candidate(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "self_judge_root"
    write_json(
        root / "base" / "instrumental_self_judge_manifest.json",
        self_judge_manifest(tmp_path, "base", minus_value=0.2, blocker_segments={1}),
    )
    write_json(
        root / "matrix_winner" / "instrumental_self_judge_manifest.json",
        self_judge_manifest(tmp_path, "matrix_winner", minus_value=0.3, blocker_segments={1, 2}),
    )
    write_json(
        root / "alt" / "instrumental_self_judge_manifest.json",
        self_judge_manifest(tmp_path, "alt", minus_value=0.8, blocker_segments=set()),
    )
    matrix_json = write_json(
        tmp_path / "matrix" / "no_reference_minus_candidate_score_matrix.json",
        {
            "summary": {
                "matrix_type": "no_reference_minus_candidate_score_matrix",
                "score_mode": "minus",
                "top_candidate_id": "matrix_winner",
                "top_next_action": "run_segment_router_for_self_blockers",
            },
            "rows": [
                {
                    "rank": 1,
                    "canonical_candidate_id": "matrix_winner",
                    "aliases": ["matrix_winner"],
                    "next_action": "run_segment_router_for_self_blockers",
                    "selection_priority": 82,
                    "result_judge": {"accepted": True, "minus_score": 0.82, "confidence": 0.8},
                    "self_judge": {"status": "proxy_blocked", "blocker_count": 2, "passed": False},
                    "delivery_allowed": False,
                    "release_ready": False,
                },
                {
                    "rank": 2,
                    "canonical_candidate_id": "base",
                    "aliases": ["base"],
                    "next_action": "blocked_generate_or_select_better_minus_candidate",
                    "selection_priority": 42,
                    "result_judge": {"accepted": False, "minus_score": 0.6, "confidence": 0.7},
                    "self_judge": {"status": "proxy_blocked", "blocker_count": 1, "passed": False},
                    "delivery_allowed": False,
                    "release_ready": False,
                },
            ],
        },
    )
    patch_accepting_gates(monkeypatch)

    assert (
        main(
            [
                "--self-judge-root",
                str(root),
                "--candidate-score-matrix-json",
                str(matrix_json),
                "--output-dir",
                str(tmp_path / "out"),
                "--crossfade-ms",
                "0",
                "--segment-sec",
                "1",
            ]
        )
        == 1
    )

    final_manifest = json.loads(
        (tmp_path / "out" / "no_reference_minus_delivery_pipeline_manifest.json").read_text(encoding="utf-8")
    )
    assert final_manifest["base_candidate_id"] == "matrix_winner"
    assert final_manifest["base_selection_source"] == "candidate_score_matrix"


def test_minus_delivery_pipeline_passes_repairable_blocked_alternate_flag(tmp_path: Path, monkeypatch) -> None:
    base_path = write_json(
        tmp_path / "base_self_judge.json",
        self_judge_manifest(tmp_path, "base", minus_value=0.2, blocker_segments={1}),
    )
    alt_path = write_json(
        tmp_path / "alt_self_judge.json",
        self_judge_manifest(tmp_path, "alt", minus_value=0.8, blocker_segments={1}),
    )
    captured = {"allow_repairable_blocked_alternate": None}

    def fake_build_minus_segment_switch_candidate(
        self_judge_payloads,
        *,
        output_dir,
        base_candidate_id,
        crossfade_ms=80,
        allow_repairable_blocked_alternate=False,
    ):
        captured["allow_repairable_blocked_alternate"] = allow_repairable_blocked_alternate
        minus_path = Path(write_wav(tmp_path / "rendered" / "minus.wav", 0.2))
        vocal_path = Path(write_wav(tmp_path / "rendered" / "vocal.wav", 0.1))
        mix_path = Path(write_wav(tmp_path / "rendered" / "mix.wav", 0.3))
        candidate_row = {
            "track_id": "track-a",
            "candidate_id": f"minus_segment_switch_{base_candidate_id}",
            "mix_path": str(mix_path),
            "source_audio_path": str(mix_path),
            "vocal_path": str(vocal_path),
            "selected_vocal_path": str(vocal_path),
            "instrumental_path": str(minus_path),
            "minus_path": str(minus_path),
            "selected_minus_path": str(minus_path),
            "requires_post_render_minus_self_judge": True,
            "can_publish_as_result": False,
            "can_mark_release_ready": False,
            "delivery_allowed": False,
        }
        candidate_row_jsonl = tmp_path / "rendered" / "candidate_row.jsonl"
        candidate_row_jsonl.write_text(json.dumps(candidate_row, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "render_status": "rendered_minus_segment_switch_candidate",
            "track_id": "track-a",
            "candidate_row_jsonl": str(candidate_row_jsonl),
            "candidate_row": candidate_row,
            "allow_repairable_blocked_alternate": allow_repairable_blocked_alternate,
        }

    monkeypatch.setattr(
        "tools.run_no_reference_minus_delivery_pipeline.build_minus_segment_switch_candidate",
        fake_build_minus_segment_switch_candidate,
    )
    patch_accepting_gates(monkeypatch)

    manifest = run_minus_delivery_pipeline(
        self_judge_manifest_paths=[base_path, alt_path],
        output_dir=tmp_path / "out",
        base_candidate_id="base",
        crossfade_ms=0,
        segment_sec=1.0,
        allow_repairable_blocked_alternate=True,
    )

    assert captured["allow_repairable_blocked_alternate"] is True
    assert_pipeline_requires_aggregate_gate(manifest)


def test_score_matrix_base_selection_prefers_specific_candidate_over_model_name_alias(tmp_path: Path) -> None:
    matrix_json = write_json(
        tmp_path / "matrix.json",
        {
            "rows": [
                {
                    "rank": 1,
                    "canonical_candidate_id": "hybrid_model_aggressive",
                    "aliases": ["hybrid_model", "hybrid_model_aggressive"],
                    "next_action": "run_segment_router_for_self_blockers",
                    "selection_priority": 82,
                }
            ]
        },
    )
    base_payload = self_judge_manifest(tmp_path, "hybrid_model", minus_value=0.2, blocker_segments={1})
    aggressive_payload = self_judge_manifest(
        tmp_path,
        "hybrid_model_aggressive",
        minus_value=0.3,
        blocker_segments={1, 2},
    )

    assert (
        choose_base_candidate_id_from_score_matrix(
            matrix_json=matrix_json,
            self_judge_payloads=[base_payload, aggressive_payload],
        )
        == "hybrid_model_aggressive"
    )


def test_score_matrix_base_selection_uses_result_judge_next_action_for_minus(tmp_path: Path) -> None:
    matrix_json = write_json(
        tmp_path / "matrix.json",
        {
            "rows": [
                {
                    "rank": 1,
                    "canonical_candidate_id": "scnet_minus",
                    "aliases": ["scnet_minus"],
                    "next_action": "run_no_reference_result_judge_minus",
                    "selection_priority": 91,
                    "self_judge": {
                        "candidate_label": "scnet_minus",
                        "status": "proxy_blocked",
                        "blocker_count": 41,
                        "passed": False,
                    },
                    "delivery_allowed": False,
                    "release_ready": False,
                },
                {
                    "rank": 2,
                    "canonical_candidate_id": "direct_minus",
                    "aliases": ["direct_minus"],
                    "next_action": "run_no_reference_result_judge_minus",
                    "selection_priority": 82,
                    "self_judge": {
                        "candidate_label": "direct_minus",
                        "status": "proxy_blocked",
                        "blocker_count": 41,
                        "passed": False,
                    },
                    "delivery_allowed": False,
                    "release_ready": False,
                },
            ]
        },
    )
    direct_payload = self_judge_manifest(tmp_path, "direct_minus", minus_value=0.2, blocker_segments={1})
    scnet_payload = self_judge_manifest(tmp_path, "scnet_minus", minus_value=0.3, blocker_segments={1, 2})

    assert (
        choose_base_candidate_id_from_score_matrix(
            matrix_json=matrix_json,
            self_judge_payloads=[direct_payload, scnet_payload],
        )
        == "scnet_minus"
    )


def test_minus_delivery_pipeline_accepts_existing_zero_blocker_candidate(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "self_judge_root"
    write_json(
        root / "clean" / "instrumental_self_judge_manifest.json",
        self_judge_manifest(tmp_path, "clean", minus_value=0.5, blocker_segments=set()),
    )
    patch_accepting_gates(monkeypatch)

    assert (
        main(
            [
                "--self-judge-root",
                str(root),
                "--output-dir",
                str(tmp_path / "out"),
                "--crossfade-ms",
                "0",
                "--segment-sec",
                "1",
            ]
        )
        == 1
    )

    final_manifest = json.loads(
        (tmp_path / "out" / "no_reference_minus_delivery_pipeline_manifest.json").read_text(encoding="utf-8")
    )
    listen_manifest = json.loads(
        (tmp_path / "out" / "no_reference_minus_listen_preview_manifest.json").read_text(encoding="utf-8")
    )
    assert final_manifest["base_candidate_id"] == "clean"
    assert final_manifest["render_status"] == "existing_minus_candidate_proxy_pass"
    assert_pipeline_requires_aggregate_gate(final_manifest)
    assert listen_manifest["delivery_status"] == "accepted"
    assert listen_manifest["delivery_allowed"] is True
    assert listen_manifest["audio_exported"] is False


def test_minus_delivery_pipeline_repairs_post_render_vocal_leakage_blocker(tmp_path: Path, monkeypatch) -> None:
    base_path = write_json(
        tmp_path / "base_self_judge.json",
        self_judge_manifest(tmp_path, "base", minus_value=0.2, blocker_segments={1}),
    )
    alt_path = write_json(
        tmp_path / "alt_self_judge.json",
        self_judge_manifest(tmp_path, "alt", minus_value=0.8, blocker_segments=set()),
    )
    judge_calls = {"count": 0}
    repair_calls = {"count": 0}

    def fake_judge_instrumental(**kwargs):
        judge_calls["count"] += 1
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = str(kwargs["candidate_instrumental_path"])
        vocal_path = str(kwargs["candidate_vocal_path"])
        blocked = judge_calls["count"] == 1
        manifest = {
            "mode": "instrumental_self_judge_gate",
            "source_audio_path": str(kwargs["source_audio_path"]),
            "candidate": {
                "label": str(kwargs["candidate_label"]),
                "instrumental_path": candidate_path,
                "vocal_path": vocal_path,
            },
            "overall": {"status": "proxy_blocked" if blocked else "proxy_pass", "blocker_count": 1 if blocked else 0},
            "segments": [
                {
                    "segment_id": "segment_0001",
                    "window_sec": [1.0, 2.0],
                    "blocker": blocked,
                    "issues": ["vocal_projection_proxy"] if blocked else [],
                    "severity_score": 0.2 if blocked else 0.0,
                }
            ],
            "delivery_contract": {"delivery_allowed": False},
        }
        write_json(output_dir / "instrumental_self_judge_manifest.json", manifest)
        return manifest

    def fake_repair(args):
        repair_calls["count"] += 1
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "minus_judge_guided_leakage_repair.wav"
        write_wav(output_path, 0.35)
        candidate_row = {
            "track_id": str(args.track_id),
            "candidate_id": f"{args.candidate_id}__judge_guided_leakage_repair",
            "mix_path": str(args.raw_audio_path),
            "source_audio_path": str(args.raw_audio_path),
            "vocal_path": str(args.primary_vocals_path),
            "selected_vocal_path": str(args.primary_vocals_path),
            "instrumental_path": str(output_path),
            "minus_path": str(output_path),
            "selected_minus_path": str(output_path),
            "candidate_family": "judge_guided_minus_vocal_leakage_repair",
            "candidate_origin": "judge_guided_projection_repair_candidate",
            "requires_post_render_minus_self_judge": True,
            "requires_dense_post_reconciliation_judge": True,
            "can_publish_as_result": False,
            "can_mark_release_ready": False,
            "delivery_allowed": False,
            "delivery_status": "blocked",
            "presentation_status": "preview_only_not_for_listen_pack",
            "release_status": "not_release_ready",
            "is_complete_pair": True,
        }
        candidate_row_jsonl = output_dir / "candidate_pair.jsonl"
        candidate_row_jsonl.write_text(json.dumps(candidate_row, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "mode": "judge_guided_minus_vocal_leakage_repair",
            "stem_type": "minus",
            "track_id": str(args.track_id),
            "candidate_id": candidate_row["candidate_id"],
            "candidate_only": True,
            "delivery_status": "candidate_only",
            "delivery_allowed": False,
            "presentation_status": "blocked_until_judged",
            "release_status": "not_release_ready",
            "listenable": False,
            "deliverable": False,
            "can_mark_release_ready": False,
            "requires_post_judge": True,
            "candidate_row_jsonl": str(candidate_row_jsonl),
            "candidate_row": candidate_row,
        }
        write_json(output_dir / "judge_guided_minus_vocal_leakage_repair_manifest.json", manifest)
        return manifest

    def fake_judge_candidate_rows(rows, *, score_mode):
        assert score_mode == "minus"
        row = rows[0]
        return {
            "rows": [
                {
                    **row,
                    "score_mode": "minus",
                    "decision": "accept_candidate",
                    "minus_decision": "accept_candidate",
                    "minus_score": 0.91,
                    "minus_blockers": [],
                    "window_count": 12,
                    "overall_no_reference_score": 0.91,
                    "confidence": 0.9,
                    "reasons": ["no_reference_proxy_passed"],
                    "minus_musical_detail_preservation_gate": {
                        "status": "proxy_pass",
                        "required_metrics": [
                            "instrumental_integrity",
                            "transient_preservation",
                            "spectral_balance",
                            "phase_stereo_stability",
                            "mixture_consistency_score",
                        ],
                        "metrics": {
                            "instrumental_integrity": 0.92,
                            "transient_preservation": 0.9,
                            "spectral_balance": 0.88,
                            "phase_stereo_stability": 0.86,
                            "mixture_consistency_score": 0.91,
                        },
                        "blockers": [],
                    },
                }
            ],
            "summary": {
                "total_tracks": 1,
                "candidates_judged": 1,
                "accepted_candidates": 1,
                "blocked_hard_case": 0,
                "usable_without_stems": 1,
                "decision_counts": {"accept_candidate": 1},
                "top_failure_reasons": {"no_reference_proxy_passed": 1},
                "safety": {},
            },
        }

    def fake_select_candidates(rows, *, score_mode):
        row = rows[0]
        return {
            "rows": [
                {
                    "track_id": row["track_id"],
                    "score_mode": score_mode,
                    "decision": "accept_candidate",
                    "selected_candidate": row,
                    "confidence": 0.9,
                    "reasons": ["selected_by_no_reference_score"],
                    "candidate_count": 1,
                    "accepted_candidate_count": 1,
                }
            ],
            "summary": {
                "score_mode": score_mode,
                "total_tracks": 1,
                "candidates_judged": 1,
                "accepted_candidates": 1,
                "selected_alternatives": 0,
                "segment_blend_needed": 0,
                "blocked_hard_case": 0,
                "usable_without_stems": 1,
                "decision_counts": {"accept_candidate": 1},
                "top_failure_reasons": {"selected_by_no_reference_score": 1},
                "safety": {},
            },
        }

    monkeypatch.setattr("tools.run_no_reference_minus_delivery_pipeline.judge_instrumental", fake_judge_instrumental)
    monkeypatch.setattr(
        "tools.run_no_reference_minus_delivery_pipeline.build_judge_guided_minus_vocal_leakage_repair",
        fake_repair,
        raising=False,
    )
    monkeypatch.setattr("tools.run_no_reference_minus_delivery_pipeline.judge_candidate_rows", fake_judge_candidate_rows)
    monkeypatch.setattr("tools.run_no_reference_minus_delivery_pipeline.select_candidates", fake_select_candidates)

    manifest = run_minus_delivery_pipeline(
        self_judge_manifest_paths=[base_path, alt_path],
        output_dir=tmp_path / "out",
        base_candidate_id="base",
        crossfade_ms=0,
        segment_sec=1.0,
    )

    assert repair_calls["count"] == 1
    assert judge_calls["count"] == 2
    assert_pipeline_requires_aggregate_gate(manifest)
    assert manifest["repair_manifest_path"].endswith("judge_guided_minus_vocal_leakage_repair_manifest.json")
    assert manifest["post_candidate_self_judge_manifest"].endswith("post_repair_self_judge/instrumental_self_judge_manifest.json")


def test_minus_delivery_pipeline_blocks_without_safe_alternate(tmp_path: Path, monkeypatch) -> None:
    base_path = write_json(
        tmp_path / "base_self_judge.json",
        self_judge_manifest(tmp_path, "base", minus_value=0.2, blocker_segments={1}),
    )
    alt_path = write_json(
        tmp_path / "alt_self_judge.json",
        self_judge_manifest(tmp_path, "alt", minus_value=0.8, blocker_segments={1}),
    )
    calls = {"self_judge": 0}

    def forbidden_judge_instrumental(**kwargs):
        calls["self_judge"] += 1
        raise AssertionError("post-render self-judge must not run when render is blocked")

    monkeypatch.setattr("tools.run_no_reference_minus_delivery_pipeline.judge_instrumental", forbidden_judge_instrumental)

    manifest = run_minus_delivery_pipeline(
        self_judge_manifest_paths=[base_path, alt_path],
        output_dir=tmp_path / "out",
        base_candidate_id="base",
        crossfade_ms=0,
        segment_sec=1.0,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    assert manifest["dominant_failure_reason"] == "no_safe_non_blocking_alternate_segment"
    assert manifest["final_listen_preview_manifest"] == ""
    assert calls["self_judge"] == 0


def test_minus_delivery_pipeline_blocks_missing_audio_before_gates(tmp_path: Path, monkeypatch) -> None:
    base_path = write_json(
        tmp_path / "base_self_judge.json",
        self_judge_manifest(tmp_path, "base", minus_value=0.2, blocker_segments={1}),
    )
    alt_path = write_json(
        tmp_path / "alt_self_judge.json",
        self_judge_manifest(tmp_path, "alt", minus_value=0.8, blocker_segments=set()),
    )
    calls = {"self_judge": 0}

    def forbidden_judge_instrumental(**kwargs):
        calls["self_judge"] += 1
        raise AssertionError("post-render gates must not run with missing audio inputs")

    monkeypatch.setattr("tools.run_no_reference_minus_delivery_pipeline.judge_instrumental", forbidden_judge_instrumental)

    manifest = run_minus_delivery_pipeline(
        self_judge_manifest_paths=[base_path, alt_path],
        output_dir=tmp_path / "out",
        base_candidate_id="base",
        source_audio_path=tmp_path / "missing_mix.wav",
        crossfade_ms=0,
        segment_sec=1.0,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "missing_audio_path_for_internal_analysis"
    assert str(tmp_path / "missing_mix.wav") in manifest["missing_audio_inputs"]
    assert calls["self_judge"] == 0


def test_minus_delivery_pipeline_cli_writes_final_manifest(tmp_path: Path, monkeypatch) -> None:
    base_path = write_json(
        tmp_path / "base_self_judge.json",
        self_judge_manifest(tmp_path, "base", minus_value=0.2, blocker_segments={1}),
    )
    alt_path = write_json(
        tmp_path / "alt_self_judge.json",
        self_judge_manifest(tmp_path, "alt", minus_value=0.8, blocker_segments={1}),
    )
    output_dir = tmp_path / "out"

    assert (
        main(
            [
                "--self-judge-manifest",
                str(base_path),
                "--self-judge-manifest",
                str(alt_path),
                "--output-dir",
                str(output_dir),
                "--base-candidate-id",
                "base",
                "--crossfade-ms",
                "0",
                "--segment-sec",
                "1",
            ]
        )
        == 1
    )
    final_manifest = output_dir / "no_reference_minus_delivery_pipeline_manifest.json"
    summary_md = output_dir / "no_reference_minus_delivery_pipeline_summary.md"
    assert final_manifest.exists()
    assert summary_md.exists()
    payload = json.loads(final_manifest.read_text(encoding="utf-8"))
    assert payload["delivery_status"] == "blocked"
    assert payload["delivery_allowed"] is False
