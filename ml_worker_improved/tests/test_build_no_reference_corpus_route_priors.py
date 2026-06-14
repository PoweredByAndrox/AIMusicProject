from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_no_reference_corpus_route_priors.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_no_reference_corpus_route_priors", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def judge_row(
    *,
    track_id: str,
    candidate_id: str,
    vocal_route_id: str = "phase_aligned_vocal",
    minus_route_id: str = "instrumental_resurrection_minus",
    vocal_decision: str = "accept_candidate",
    minus_decision: str = "blocked_hard_case",
    pair_decision: str = "pair_compatible",
    reasons: list[str] | None = None,
    minus_blockers: list[str] | None = None,
    vocal_score: float = 0.80,
    minus_score: float = 0.64,
    pair_score: float = 0.78,
    vocal_leakage: float = 0.88,
    split: str = "train",
) -> dict[str, object]:
    return {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "score_mode": "both",
        "vocal_route_id": vocal_route_id,
        "minus_route_id": minus_route_id,
        "vocal_path": f"/app/artifacts/{track_id}/vocal/{vocal_route_id}/vocals.flac",
        "minus_path": f"/app/artifacts/{track_id}/minus/{minus_route_id}/minus.flac",
        "instrumental_path": f"/app/artifacts/{track_id}/minus/{minus_route_id}/minus.flac",
        "vocal_decision": vocal_decision,
        "minus_decision": minus_decision,
        "pair_decision": pair_decision,
        "decision": "blocked_hard_case" if minus_decision == "blocked_hard_case" else "accept_candidate",
        "reasons": reasons or ["vocal_leakage_in_minus"],
        "minus_blockers": minus_blockers or ["severe_vocal_bleed_in_minus"],
        "vocal_blockers": [],
        "pair_blockers": [],
        "vocal_score": vocal_score,
        "minus_score": minus_score,
        "pair_compatibility_score": pair_score,
        "vocal_leakage_in_minus_score": vocal_leakage,
        "mixture_consistency_score": 0.90,
        "window_count": 5,
        "evaluation_split": split,
    }


def route_row(result: dict[str, object], stem_type: str, route_id: str) -> dict[str, object]:
    rows = result["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["stem_type"] == stem_type and row["route_id"] == route_id)


def test_repeated_minus_leakage_blocks_route_without_training() -> None:
    module = _load_module()
    rows = [
        judge_row(track_id="track-a", candidate_id="a"),
        judge_row(track_id="track-b", candidate_id="b"),
    ]

    result = module.build_corpus_route_priors(rows, source_paths=["judge_a.jsonl"], min_evidence=2)
    minus = route_row(result, "minus", "instrumental_resurrection_minus")
    vocal = route_row(result, "vocal", "phase_aligned_vocal")

    assert minus["candidate_count"] == 2
    assert minus["accepted_count"] == 0
    assert minus["blocked_count"] == 2
    assert minus["recommendation"] == "blocked_as_minus_baseline"
    assert minus["requires_new_source_candidate"] is True
    assert "severe_vocal_bleed_in_minus" in minus["top_blockers"]
    assert vocal["accepted_count"] == 2
    assert vocal["recommendation"] == "keep_as_vocal_candidate_source"
    assert result["manifest"]["safety"]["training_started"] is False
    assert result["manifest"]["safety"]["fine_tune_started"] is False
    assert result["manifest"]["delivery_allowed"] is False
    assert result["manifest"]["release_status"] == "not_release_ready"


def test_vocal_and_minus_evidence_stays_mode_specific() -> None:
    module = _load_module()
    rows = [
        judge_row(track_id="track-a", candidate_id="good-vocal-bad-minus", pair_decision="pair_compatible"),
        judge_row(
            track_id="track-b",
            candidate_id="bad-vocal-bad-minus",
            vocal_decision="blocked_hard_case",
            minus_decision="blocked_hard_case",
            pair_decision="blocked_pair_incompatible",
            reasons=["phase_or_stereo_inconsistency", "vocal_leakage_in_minus"],
            vocal_score=0.58,
            minus_score=0.61,
            pair_score=0.50,
        ),
    ]

    result = module.build_corpus_route_priors(rows, source_paths=["judge.jsonl"], min_evidence=1)
    summary = result["manifest"]["summary"]

    assert summary["vocal_accept_minus_blocked_count"] == 1
    assert summary["pair_compatible_minus_blocked_count"] == 1
    assert route_row(result, "vocal", "phase_aligned_vocal")["accepted_count"] == 1
    minus = route_row(result, "minus", "instrumental_resurrection_minus")
    assert minus["blocked_count"] == 2
    assert minus["recommendation"] == "blocked_as_minus_baseline"
    assert minus["requires_new_source_candidate"] is True


def test_minus_mode_pair_incompatible_row_blocks_minus_route_prior() -> None:
    module = _load_module()
    row = judge_row(
        track_id="track-a",
        candidate_id="pair-blocked-minus",
        minus_decision="accept_candidate",
        pair_decision="blocked_pair_incompatible",
        reasons=["phase_or_stereo_inconsistency"],
        minus_blockers=[],
        minus_score=0.878842,
        pair_score=0.751346,
        vocal_leakage=0.277375,
    )
    row["score_mode"] = "minus"
    row["pair_blockers"] = ["phase_collapse"]

    result = module.build_corpus_route_priors([row], source_paths=["judge.jsonl"], min_evidence=1)
    minus = route_row(result, "minus", "instrumental_resurrection_minus")

    assert minus["accepted_count"] == 0
    assert minus["blocked_count"] == 1
    assert minus["recommendation"] == "blocked_as_minus_baseline"
    assert minus["requires_new_source_candidate"] is True
    assert "phase_collapse" in minus["top_blockers"]
    assert "pair_incompatible" in minus["top_failure_reasons"]


def test_result_accepted_matrix_row_with_self_judge_blockers_blocks_minus_route_prior() -> None:
    module = _load_module()
    row = {
        "track_id": "track-a",
        "canonical_candidate_id": "track-a__htdemucs_6s",
        "target_stem": "minus",
        "model_name": "htdemucs_6s",
        "minus_path": "/app/artifacts/track-a/htdemucs_6s/minus.flac",
        "vocal_path": "/app/artifacts/track-a/htdemucs_6s/vocals.flac",
        "result_judge": {
            "accepted": True,
            "candidate_id": "manifest_0001",
            "decision": "accept_candidate",
            "score_mode": "minus",
            "minus_score": 0.940063,
            "minus_blockers": [],
            "reasons": ["no_reference_proxy_passed"],
            "vocal_leakage_in_minus_score": 0.221,
            "mixture_consistency_score": 0.91,
        },
        "self_judge": {
            "passed": False,
            "status": "proxy_blocked",
            "blocker_count": 27,
            "blocker_ratio": 0.6,
            "candidate_label": "htdemucs_6s_candidate_wave_minus",
            "top_blockers": [
                {
                    "issues": ["vocal_stem_corr_proxy", "vocal_projection_proxy"],
                    "severity_score": 1.964511,
                }
            ],
        },
    }

    result = module.build_corpus_route_priors([row], source_paths=["matrix.jsonl"], min_evidence=1)
    minus = route_row(result, "minus", "htdemucs_6s")

    assert minus["accepted_count"] == 0
    assert minus["blocked_count"] == 1
    assert minus["recommendation"] == "blocked_as_minus_baseline"
    assert minus["requires_new_source_candidate"] is True
    assert minus["source_pool_eligible"] is False
    assert minus["decision_counts"] == {"blocked_self_judge": 1}
    assert "self_judge_proxy_blocked" in minus["top_failure_reasons"]
    assert "vocal_stem_corr_proxy" in minus["top_blockers"]
    assert "vocal_projection_proxy" in minus["top_blockers"]


def test_derived_minus_route_alias_blocks_executable_model_route() -> None:
    module = _load_module()
    row = judge_row(
        track_id="track-a",
        candidate_id="htdemucs-ft-derived-minus",
        minus_route_id="htdemucs_ft_derived_non_vocal_sum",
    )
    row["minus_path"] = "/app/artifacts/track-a/minus/htdemucs_ft_derived_non_vocal_sum/minus.flac"
    row["instrumental_path"] = row["minus_path"]
    rows = [row]

    result = module.build_corpus_route_priors(rows, source_paths=["judge.jsonl"], min_evidence=1)
    minus = route_row(result, "minus", "htdemucs_ft")

    assert minus["candidate_count"] == 1
    assert minus["recommendation"] == "blocked_as_minus_baseline"
    assert minus["requires_new_source_candidate"] is True


def test_minus_source_pool_evidence_survives_pair_incompatible_rows() -> None:
    module = _load_module()
    source_row = judge_row(
        track_id="track-a",
        candidate_id="becruily_source_pair",
        vocal_route_id="becruily_residual_vocal",
        minus_route_id="becruily_instrumental_vocals",
        vocal_decision="blocked_hard_case",
        minus_decision="blocked_hard_case",
        pair_decision="blocked_pair_incompatible",
        reasons=["instrumental_leakage_in_vocal", "phase_or_stereo_inconsistency"],
        minus_blockers=["severe_phase_collapse"],
        vocal_score=0.59,
        minus_score=0.832581,
        pair_score=0.70,
        vocal_leakage=0.284465,
    )
    source_row["mixture_consistency_score"] = 0.929946
    cross_bad_rows = [
        judge_row(
            track_id="track-a",
            candidate_id="cross_bad_a",
            vocal_route_id="htdemucs_6s",
            minus_route_id="becruily_instrumental_vocals",
            vocal_decision="blocked_hard_case",
            minus_decision="blocked_hard_case",
            pair_decision="blocked_pair_incompatible",
            reasons=["bad_mixture_reconstruction", "vocal_leakage_in_minus", "phase_or_stereo_inconsistency"],
            minus_blockers=["severe_vocal_bleed_in_minus", "severe_music_damage", "severe_phase_collapse"],
            minus_score=0.305067,
            pair_score=0.0,
            vocal_leakage=0.998216,
        ),
        judge_row(
            track_id="track-b",
            candidate_id="cross_bad_b",
            vocal_route_id="htdemucs_ft",
            minus_route_id="becruily_instrumental_vocals",
            vocal_decision="blocked_hard_case",
            minus_decision="blocked_hard_case",
            pair_decision="blocked_pair_incompatible",
            reasons=["bad_mixture_reconstruction", "vocal_leakage_in_minus", "silence_hallucination"],
            minus_blockers=["severe_vocal_bleed_in_minus", "severe_music_damage", "severe_phase_collapse"],
            minus_score=0.301579,
            pair_score=0.0,
            vocal_leakage=0.982624,
        ),
    ]

    result = module.build_corpus_route_priors([source_row, *cross_bad_rows], source_paths=["judge.jsonl"], min_evidence=1)
    minus = route_row(result, "minus", "becruily_instrumental_vocals")

    assert minus["recommendation"] == "blocked_as_minus_baseline"
    assert minus["requires_new_source_candidate"] is True
    assert minus["source_pool_eligible"] is True
    assert minus["source_pool_status"] == "minus_source_candidate_supported_for_repair"
    assert minus["best_source_track_id"] == "track-a"
    assert minus["best_source_candidate_id"] == "becruily_source_pair"
    assert minus["best_source_score"] == 0.832581
    assert minus["best_source_vocal_leakage_in_minus_score"] == 0.284465
    assert "low_vocal_leakage_minus_source_available" in minus["source_pool_reasons"]
    assert minus["delivery_allowed"] is False
    assert minus["release_status"] == "not_release_ready"


def test_holdout_rows_are_reported_but_do_not_drive_train_priors() -> None:
    module = _load_module()
    rows = [
        judge_row(track_id="track-a", candidate_id="train-a"),
        judge_row(track_id="track-h", candidate_id="holdout-a", split="holdout"),
    ]

    result = module.build_corpus_route_priors(rows, source_paths=["judge.jsonl"], min_evidence=2)
    minus = route_row(result, "minus", "instrumental_resurrection_minus")

    assert minus["candidate_count"] == 1
    assert minus["holdout_candidate_count"] == 1
    assert minus["recommendation"] == "insufficient_evidence"
    assert result["manifest"]["summary"]["holdout_rows_skipped_for_priors"] == 1


def test_write_outputs_are_deterministic_and_metadata_only(tmp_path: Path) -> None:
    module = _load_module()
    rows = [
        judge_row(track_id="track-b", candidate_id="b"),
        judge_row(track_id="track-a", candidate_id="a"),
    ]
    result = module.build_corpus_route_priors(rows, source_paths=["judge.jsonl"], min_evidence=2)
    output_jsonl = tmp_path / "route_priors.jsonl"
    manifest_json = tmp_path / "route_priors_manifest.json"
    summary_md = tmp_path / "route_priors_summary.md"

    module.write_outputs(result, output_jsonl=output_jsonl, manifest_json=manifest_json, summary_md=summary_md)
    first_jsonl = output_jsonl.read_text(encoding="utf-8")
    module.write_outputs(result, output_jsonl=output_jsonl, manifest_json=manifest_json, summary_md=summary_md)
    second_jsonl = output_jsonl.read_text(encoding="utf-8")
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))

    assert first_jsonl == second_jsonl
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "route_prior_not_listen_pack"
    assert not list(tmp_path.rglob("*.wav"))
    assert not list(tmp_path.rglob("*.flac"))
    assert "- route_rows:" in summary_md.read_text(encoding="utf-8")


def test_cli_uses_docker_guard_and_writes_outputs(tmp_path: Path) -> None:
    module = _load_module()
    judge_jsonl = tmp_path / "judge.jsonl"
    judge_jsonl.write_text(json.dumps(judge_row(track_id="track-a", candidate_id="a")) + "\n", encoding="utf-8")
    output_jsonl = tmp_path / "route_priors.jsonl"
    manifest_json = tmp_path / "route_priors_manifest.json"
    summary_md = tmp_path / "route_priors_summary.md"

    with patch.object(module, "require_docker_runtime") as guard:
        exit_code = module.main(
            [
                "--judge-jsonl",
                str(judge_jsonl),
                "--output-jsonl",
                str(output_jsonl),
                "--manifest-json",
                str(manifest_json),
                "--summary-md",
                str(summary_md),
                "--min-evidence",
                "1",
            ],
        )

    assert exit_code == 0
    guard.assert_called_once()
    assert output_jsonl.exists()
    assert manifest_json.exists()
    assert summary_md.exists()
