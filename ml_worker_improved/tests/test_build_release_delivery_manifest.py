from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools import build_release_delivery_manifest as delivery


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_delivery_manifest_keeps_release_ready_as_deliverable(tmp_path: Path) -> None:
    batch_summary_path = tmp_path / "batch_summary.json"
    _write_json(
        batch_summary_path,
        {
            "mode": "adaptive_vocal_contract_batch",
            "track_count": 1,
            "tracks": [
                {
                    "source_path": "/music/source.mp3",
                    "manifest_path": "/app/out/adaptive_vocal_manifest.json",
                    "selected_profile": "clean",
                    "selected_score": 1.2,
                    "exit_code": 0,
                    "release_status": "release_ready",
                    "delivery_status": "deliverable",
                    "deliverable_vocals_flac": "/app/out/01_adaptive_selected_vocals.flac",
                    "deliverable_listen_vocals_flac": "/app/listen/01_adaptive_selected_vocals.flac",
                    "diagnostic_vocals_flac": None,
                    "diagnostic_listen_vocals_flac": None,
                    "blocker_reasons": [],
                    "blocker_labels": [],
                    "worst_windows": [],
                }
            ],
        },
    )

    manifest = delivery.build_release_delivery_manifest(
        batch_summary_path=batch_summary_path,
        output_path=tmp_path / "release_delivery_manifest.json",
    )

    assert manifest["deliverable_count"] == 1
    assert manifest["diagnostic_count"] == 0
    track = manifest["tracks"][0]
    assert track["delivery_status"] == "deliverable"
    assert track["deliverable"]["vocals_flac"] == "/app/out/01_adaptive_selected_vocals.flac"
    assert track["diagnostic"]["vocals_flac"] is None


def test_delivery_manifest_blocks_legacy_track_without_deliverable_fields(tmp_path: Path) -> None:
    batch_summary_path = tmp_path / "legacy_batch_summary.json"
    _write_json(
        batch_summary_path,
        {
            "mode": "adaptive_vocal_contract_batch",
            "tracks": [
                {
                    "source_path": "/music/source.mp3",
                    "selected_profile": "legacy",
                    "exit_code": 0,
                    "release_status": "release_ready",
                    "selected_vocals_flac": "/app/out/01_adaptive_selected_vocals.flac",
                    "listen_selected_vocals_flac": "/app/listen/01_adaptive_selected_vocals.flac",
                }
            ],
        },
    )

    manifest = delivery.build_release_delivery_manifest(
        batch_summary_path=batch_summary_path,
        output_path=tmp_path / "release_delivery_manifest.json",
    )

    track = manifest["tracks"][0]
    assert manifest["deliverable_count"] == 0
    assert manifest["diagnostic_count"] == 1
    assert track["delivery_status"] == "diagnostic_only"
    assert track["release_status"] == "diagnostic_only"
    assert track["deliverable"]["vocals_flac"] is None
    assert track["diagnostic"]["vocals_flac"] == "/app/out/01_adaptive_selected_vocals.flac"
    assert track["blockers"]["reasons"] == ["missing_delivery_gate"]


def test_cli_script_path_writes_release_delivery_manifest(tmp_path: Path) -> None:
    batch_summary_path = tmp_path / "batch_summary.json"
    output_path = tmp_path / "release_delivery_manifest.json"
    _write_json(
        batch_summary_path,
        {
            "mode": "adaptive_vocal_contract_batch",
            "tracks": [
                {
                    "source_path": "/music/source.mp3",
                    "exit_code": 1,
                    "contract_status": "route_failed",
                }
            ],
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(delivery.__file__)),
            str(batch_summary_path),
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["failed_count"] == 1
    assert manifest["tracks"][0]["delivery_status"] == "route_failed"


def test_delivery_manifest_treats_string_exit_code_as_route_failed(tmp_path: Path) -> None:
    track = delivery.build_delivery_track(
        {
            "source_path": "/music/source.mp3",
            "exit_code": "1",
            "release_status": "release_ready",
            "delivery_status": "deliverable",
            "deliverable_vocals_flac": "/app/out/01_adaptive_selected_vocals.flac",
        }
    )

    assert track["delivery_status"] == "route_failed"
    assert track["release_status"] == "route_failed"
    assert track["deliverable"]["vocals_flac"] is None
    assert track["blockers"]["reasons"] == ["route_failed"]
