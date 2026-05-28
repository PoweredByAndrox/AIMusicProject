from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_acceptance_first_vocal_ladder as ladder


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def registry(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = SimpleNamespace(model_id="known_vocal_model", aliases=("known_alias",))
    monkeypatch.setattr(ladder, "default_model_registry", lambda: SimpleNamespace(entries=(entry,)))


def test_rejected_candidate_is_never_exported_as_listen_result(tmp_path: Path) -> None:
    rejected = write_json(
        tmp_path / "rejected.json",
        {
            "model_id": "known_vocal_model",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "delivery_status": "rejected",
        },
    )
    listen_dir = tmp_path / "listen"
    manifest = ladder.build_final_manifest([rejected], listen_dir=listen_dir)
    assert manifest["status"] == "no_accepted_candidate"
    assert manifest["delivery_allowed"] is False
    assert manifest["selected_candidate"] is None
    assert not listen_dir.exists()


def test_candidate_space_exhausted_never_triggers_cleanup_debleed(tmp_path: Path) -> None:
    rejected = write_json(
        tmp_path / "exhausted.json",
        {
            "model_id": "known_vocal_model",
            "delivery_allowed": False,
            "delivery_status": "candidate_space_exhausted",
        },
    )
    manifest = ladder.build_final_manifest([rejected])
    assert manifest["status"] == "no_accepted_candidate"
    assert "no_cleanup_debleed_denoise_escalation" in manifest["forbidden_actions_enforced"]


def test_unregistered_model_id_is_rejected(tmp_path: Path) -> None:
    unknown = write_json(
        tmp_path / "unknown.json",
        {
            "model_id": "not_registered",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
        },
    )
    manifest = ladder.build_final_manifest([unknown])
    assert manifest["status"] == "no_accepted_candidate"
    assert manifest["evaluated_candidates"][0]["registration_reason"] == "unregistered_model_id:not_registered"


def test_registered_alias_is_allowed(tmp_path: Path) -> None:
    alias = write_json(
        tmp_path / "alias.json",
        {
            "model_id": "known_alias",
            "delivery_allowed": False,
            "delivery_status": "rejected",
        },
    )
    manifest = ladder.build_final_manifest([alias])
    assert manifest["evaluated_candidates"][0]["registered"] is True


def test_accepted_candidate_can_be_exported_only_with_listenable_status(tmp_path: Path) -> None:
    audio = tmp_path / "candidate.flac"
    audio.write_bytes(b"fake-audio")
    accepted = write_json(
        tmp_path / "accepted.json",
        {
            "model_id": "known_vocal_model",
            "candidate_id": "c1",
            "delivery_allowed": True,
            "delivery_status": "accepted",
            "presentation_status": "accepted_for_listen_pack",
            "audio_path": str(audio),
        },
    )
    listen_dir = tmp_path / "listen"
    manifest = ladder.build_final_manifest([accepted], listen_dir=listen_dir)
    assert manifest["status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert Path(manifest["exported_listen_audio"]).exists()


def test_accepted_looking_delivery_disallowed_candidate_is_not_exported(tmp_path: Path) -> None:
    audio = tmp_path / "candidate.flac"
    audio.write_bytes(b"fake-audio")
    blocked = write_json(
        tmp_path / "blocked.json",
        {
            "model_id": "known_vocal_model",
            "candidate_id": "c1",
            "delivery_allowed": False,
            "delivery_status": "accepted",
            "presentation_status": "accepted_for_listen_pack",
            "audio_path": str(audio),
        },
    )
    listen_dir = tmp_path / "listen"
    manifest = ladder.build_final_manifest([blocked], listen_dir=listen_dir)
    assert manifest["status"] == "no_accepted_candidate"
    assert manifest["selected_candidate"] is None
    assert not listen_dir.exists()


def test_no_accepted_candidate_produces_blocked_manifest(tmp_path: Path) -> None:
    rejected = write_json(
        tmp_path / "rejected.json",
        {"model_id": "known_vocal_model", "delivery_allowed": False, "delivery_status": "rejected"},
    )
    manifest = ladder.build_final_manifest([rejected])
    assert manifest["status"] == "no_accepted_candidate"
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    assert manifest["dominant_failure_reason"]
