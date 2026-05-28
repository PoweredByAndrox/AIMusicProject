from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.artifact_status_contract import (
    ArtifactContractError,
    ArtifactStatusView,
    can_promote_to_listen_result,
    stable_track_hash,
    stable_track_split,
    status_view_from_manifest,
)


def write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_rejected_candidate_is_never_promotable_as_listen_result(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "candidate.json",
        {
            "delivery_status": "rejected",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "dominant_failure_reason": "low_sdr",
        },
    )
    with pytest.raises(ArtifactContractError):
        can_promote_to_listen_result(manifest)


def test_diagnostic_blocked_artifact_is_never_result(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "diagnostic.json",
        {
            "delivery_status": "diagnostic_only",
            "delivery_allowed": False,
            "presentation_status": "diagnostic_only",
            "dominant_failure_reason": "artifact_or_timbre_damage",
        },
    )
    view = status_view_from_manifest(manifest)
    assert view.blocked
    assert not view.listenable
    with pytest.raises(ArtifactContractError):
        view.assert_promotable()


def test_accepted_for_listen_pack_is_listenable_when_delivery_allowed(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "accepted.json",
        {
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
        },
    )
    view = status_view_from_manifest(manifest)
    assert view.listenable
    view.assert_promotable()


def test_release_ready_is_deliverable_when_delivery_allowed(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "release.json",
        {
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_status": "release_ready",
        },
    )
    view = status_view_from_manifest(manifest)
    assert view.deliverable
    view.assert_promotable()


def test_delivery_allowed_false_overrides_accepted_for_listen_pack(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "inconsistent.json",
        {
            "delivery_status": "accepted",
            "delivery_allowed": False,
            "presentation_status": "accepted_for_listen_pack",
        },
    )
    view = status_view_from_manifest(manifest)
    assert view.blocked
    assert not view.listenable
    assert view.dominant_failure_reason == "unknown_blocked_reason"
    with pytest.raises(ArtifactContractError):
        view.assert_promotable()


def test_delivery_allowed_false_overrides_release_ready(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "release_blocked.json",
        {
            "delivery_status": "accepted",
            "delivery_allowed": False,
            "release_status": "release_ready",
        },
    )
    view = status_view_from_manifest(manifest)
    assert view.blocked
    assert not view.deliverable
    with pytest.raises(ArtifactContractError):
        view.assert_promotable()


def test_plain_accepted_presentation_status_is_not_listenable(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "plain_accepted.json",
        {
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted",
        },
    )
    view = status_view_from_manifest(manifest)
    assert not view.listenable
    assert not view.deliverable
    with pytest.raises(ArtifactContractError):
        view.assert_promotable()


def test_artifact_without_manifest_provenance_cannot_be_promoted() -> None:
    view = ArtifactStatusView(
        delivery_status="accepted",
        delivery_allowed=True,
        presentation_status="accepted_for_listen_pack",
        release_status="accepted_for_listen_pack",
        source_manifest="",
        manifest_sha256="",
        dominant_failure_reason="none",
        stem_type="vocal",
    )
    with pytest.raises(ArtifactContractError):
        view.assert_promotable()


def test_track_split_is_stable_by_track_hash_not_window() -> None:
    track_hash = stable_track_hash("track-a")
    assert stable_track_split(track_hash) == stable_track_split(track_hash)
