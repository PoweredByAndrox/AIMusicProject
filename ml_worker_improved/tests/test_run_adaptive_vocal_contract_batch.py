from __future__ import annotations

import json
from pathlib import Path

from tools import run_adaptive_vocal_contract_batch as batch


def test_slugify_keeps_batch_output_names_stable() -> None:
    assert batch.slugify("Nine Thou (Grant Mohrman Superstars Remix) (320)") == "nine_thou_grant_mohrman_superstars_remix_320"


def test_contract_summary_from_manifest_extracts_issue_contract_fields() -> None:
    manifest = {
        "selected_profile": "segment_issue_repair",
        "selected_score": 1.25,
        "detected_issue_action_contract": {
            "status": "uncovered_labels",
            "summary": {
                "issue_segment_count": 2,
                "covered_labels": ["vocal_dropout"],
                "rejected_labels": ["backing_vocal_loss_risk"],
                "uncovered_labels": ["new_label"],
            },
        },
        "selected_artifacts": {
            "selected_vocals_flac": "/app/out/01_adaptive_selected_vocals.flac",
            "listen_selected_vocals_flac": "/app/listen/01_adaptive_selected_vocals.flac",
            "listen_segment_issue_clips_manifest": "/app/listen/segment_issues/manifest.json",
            "release_status": "release_ready",
        },
        "defect_effectiveness_gate": {
            "status": "release_ready",
            "release_status": "release_ready",
            "blockers": [],
        },
        "issue_effectiveness_ledger": {
            "summary": {"blocked_window_count": 0},
            "windows": [],
        },
    }

    summary = batch.contract_summary_from_manifest(manifest)

    assert summary["selected_profile"] == "segment_issue_repair"
    assert summary["contract_status"] == "uncovered_labels"
    assert summary["uncovered_labels"] == ["new_label"]
    assert summary["issue_segment_count"] == 2
    assert summary["listen_segment_issue_clips_manifest"] == "/app/listen/segment_issues/manifest.json"
    assert summary["release_status"] == "release_ready"
    assert summary["artifact_release_status"] == "release_ready"
    assert summary["release_status_drift"] is False
    assert summary["delivery_status"] == "deliverable"
    assert summary["deliverable_vocals_flac"] == "/app/out/01_adaptive_selected_vocals.flac"
    assert summary["diagnostic_vocals_flac"] is None


def test_contract_summary_blocks_diagnostic_vocal_from_deliverable_paths() -> None:
    manifest = {
        "selected_profile": "candidate_16_bs_artifact_clean",
        "selected_score": 0.91,
        "detected_issue_action_contract": {
            "status": "pass",
            "summary": {"issue_segment_count": 1, "uncovered_labels": []},
        },
        "defect_effectiveness_gate": {
            "status": "blocked_for_listening",
            "release_status": "diagnostic_only",
            "blockers": [
                {
                    "reason": "selected_candidate_still_has_severe_issue",
                    "labels": ["vocal_dropout", "high_noise_or_robotic_artifact"],
                }
            ],
        },
        "issue_effectiveness_ledger": {
            "summary": {"blocked_window_count": 1},
            "windows": [
                {
                    "start_sec": 25.0,
                    "end_sec": 30.0,
                    "severity_score": 1.12,
                    "labels": ["vocal_dropout"],
                    "release_decision": "blocked",
                }
            ],
        },
        "selected_artifacts": {
            "selected_vocals_flac": "/app/out/01_adaptive_selected_vocals.flac",
            "listen_selected_vocals_flac": "/app/listen/01_adaptive_selected_vocals.flac",
            "listen_segment_issue_clips_manifest": "/app/listen/segment_issues/manifest.json",
            "release_status": "diagnostic_only",
        },
    }

    summary = batch.contract_summary_from_manifest(manifest)

    assert summary["release_status"] == "diagnostic_only"
    assert summary["artifact_release_status"] == "diagnostic_only"
    assert summary["release_status_drift"] is False
    assert summary["delivery_status"] == "diagnostic_only"
    assert summary["deliverable_vocals_flac"] is None
    assert summary["deliverable_listen_vocals_flac"] is None
    assert summary["diagnostic_vocals_flac"] == "/app/out/01_adaptive_selected_vocals.flac"
    assert summary["diagnostic_listen_vocals_flac"] == "/app/listen/01_adaptive_selected_vocals.flac"
    assert summary["blocker_reasons"] == ["selected_candidate_still_has_severe_issue"]
    assert summary["blocker_labels"] == ["high_noise_or_robotic_artifact", "vocal_dropout"]
    assert summary["blocked_window_count"] == 1
    assert summary["worst_windows"] == [
        {
            "start_sec": 25.0,
            "end_sec": 30.0,
            "severity_score": 1.12,
            "labels": ["vocal_dropout"],
            "release_decision": "blocked",
        }
    ]


def test_contract_summary_reports_release_status_drift() -> None:
    manifest = {
        "selected_profile": "candidate_16_bs_artifact_clean",
        "selected_score": 0.91,
        "detected_issue_action_contract": {"status": "pass", "summary": {}},
        "defect_effectiveness_gate": {
            "status": "blocked_for_listening",
            "release_status": "diagnostic_only",
            "blockers": [{"reason": "selected_candidate_still_has_clustered_issues"}],
        },
        "selected_artifacts": {
            "selected_vocals_flac": "/app/out/01_adaptive_selected_vocals.flac",
            "listen_selected_vocals_flac": "/app/listen/01_adaptive_selected_vocals.flac",
            "release_status": "release_ready",
        },
    }

    summary = batch.contract_summary_from_manifest(manifest)

    assert summary["release_status"] == "diagnostic_only"
    assert summary["artifact_release_status"] == "release_ready"
    assert summary["release_status_drift"] is True
    assert summary["deliverable_vocals_flac"] is None


def test_contract_summary_never_delivers_without_release_gate() -> None:
    manifest = {
        "selected_profile": "legacy_candidate",
        "selected_score": 0.9,
        "detected_issue_action_contract": {"status": "pass", "summary": {}},
        "selected_artifacts": {
            "selected_vocals_flac": "/app/out/01_adaptive_selected_vocals.flac",
            "listen_selected_vocals_flac": "/app/listen/01_adaptive_selected_vocals.flac",
            "release_status": "release_ready",
        },
    }

    summary = batch.contract_summary_from_manifest(manifest)

    assert summary["gate_status"] == "missing_release_gate"
    assert summary["release_status"] == "diagnostic_only"
    assert summary["artifact_release_status"] == "release_ready"
    assert summary["release_status_drift"] is True
    assert summary["deliverable_vocals_flac"] is None
    assert summary["diagnostic_vocals_flac"] == "/app/out/01_adaptive_selected_vocals.flac"


def test_main_writes_summary_and_continues_after_route_failure(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "track one.mp3").write_bytes(b"fake")
    output_root = tmp_path / "out"
    listen_root = tmp_path / "listen"
    summary_path = tmp_path / "summary.json"

    def fake_run_track(**kwargs):
        return {
            "source_path": str(kwargs["source_path"]),
            "output_dir": str(kwargs["output_dir"]),
            "listen_dir": str(kwargs["listen_dir"]),
            "manifest_path": str(kwargs["output_dir"] / "adaptive_vocal_manifest.json"),
            "exit_code": 1,
            "elapsed_sec": 0.1,
            "contract_status": "route_failed",
            "uncovered_labels": [],
        }

    monkeypatch.setattr(batch, "run_track", fake_run_track)

    exit_code = batch.main(
        [
            str(input_dir),
            str(output_root),
            "--listen-root",
            str(listen_root),
            "--summary-path",
            str(summary_path),
        ]
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert summary["track_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["release_ready_count"] == 0
    assert summary["diagnostic_only_count"] == 0
    assert summary["release_blocked_count"] == 0
    assert summary["tracks"][0]["contract_status"] == "route_failed"
