from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import tools.build_testing_stuff_music_corpus_plan as corpus_plan
from tools.build_testing_stuff_music_corpus_plan import DEFAULT_TARGET_STEMS, build_corpus_plan, write_outputs


def touch_audio(path: Path, payload: bytes = b"placeholder") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def rows_by_track(result: dict) -> dict[str, dict]:
    return {row["track_id"]: row for row in result["rows"]}


def test_builds_docker_visible_evaluation_queue_without_training(tmp_path: Path) -> None:
    source_root = tmp_path / "testing_stuff_music"
    first = touch_audio(source_root / "Artist - Song One.mp3", b"mp3")
    second = touch_audio(source_root / "nested" / "Song Two.flac", b"flac")
    ignored = touch_audio(source_root / "notes.txt", b"text")
    assert ignored.exists()

    result = build_corpus_plan(
        source_root=source_root,
        docker_root_label="/host_testing_stuff_music",
        host_root_label="D:/AIMusicProject/testing_stuff_music",
    )

    assert result["summary"]["corpus_status"] == "ready"
    assert result["summary"]["total_audio_files"] == 2
    assert result["summary"]["evaluation_queue_count"] == 2
    assert result["summary"]["target_stems"] == list(DEFAULT_TARGET_STEMS)
    assert result["summary"]["training_allowed_count"] == 0
    assert result["summary"]["fine_tune_allowed_count"] == 0
    assert result["summary"]["audio_read"] is False
    assert result["summary"]["audio_copied"] is False
    assert result["summary"]["audio_exported"] is False
    assert result["summary"]["models_changed"] is False

    rows = rows_by_track(result)
    assert sorted(rows) == ["artist_song_one", "song_two"]
    assert rows["artist_song_one"]["source_audio_path"] == str(first)
    assert rows["artist_song_one"]["docker_audio_path"] == "/host_testing_stuff_music/Artist - Song One.mp3"
    assert rows["artist_song_one"]["host_audio_path_hint"] == "D:/AIMusicProject/testing_stuff_music/Artist - Song One.mp3"
    assert rows["song_two"]["docker_audio_path"] == "/host_testing_stuff_music/nested/Song Two.flac"
    assert rows["song_two"]["host_audio_path_hint"] == "D:/AIMusicProject/testing_stuff_music/nested/Song Two.flac"
    assert all(row["can_start_training_now"] is False for row in result["rows"])
    assert all(row["target_stems"] == list(DEFAULT_TARGET_STEMS) for row in result["rows"])
    assert all(row["reference_available"] is False for row in result["rows"])
    assert all(row["evaluation_allowed"] is True for row in result["rows"])
    assert all("no_training_start" in row["forbidden_actions"] for row in result["rows"])


def test_missing_source_root_blocks_plan_without_fake_queue(tmp_path: Path) -> None:
    result = build_corpus_plan(source_root=tmp_path / "missing")

    assert result["rows"] == []
    assert result["summary"]["corpus_status"] == "blocked_missing_source_root"
    assert result["summary"]["evaluation_queue_count"] == 0
    assert result["manifest"]["delivery_allowed"] is False
    assert result["manifest"]["training_started"] is False
    assert result["manifest"]["release_ready"] is False


def test_max_tracks_caps_queue_deterministically(tmp_path: Path) -> None:
    source_root = tmp_path / "testing_stuff_music"
    touch_audio(source_root / "b.mp3")
    touch_audio(source_root / "a.mp3")
    touch_audio(source_root / "c.mp3")

    result = build_corpus_plan(source_root=source_root, max_tracks=2)

    assert [row["track_id"] for row in result["rows"]] == ["a", "b"]
    assert result["summary"]["total_audio_files"] == 3
    assert result["summary"]["evaluation_queue_count"] == 2
    assert result["summary"]["capped_audio_files"] == 1


def test_write_outputs_creates_only_plan_artifacts(tmp_path: Path) -> None:
    source_root = tmp_path / "testing_stuff_music"
    touch_audio(source_root / "song.mp3")
    result = build_corpus_plan(source_root=source_root)
    output_dir = tmp_path / "out"

    write_outputs(result, output_dir)

    assert (output_dir / "testing_stuff_music_corpus_plan.jsonl").exists()
    assert (output_dir / "testing_stuff_music_corpus_manifest.json").exists()
    assert (output_dir / "testing_stuff_music_corpus_summary.md").exists()
    assert not list(output_dir.rglob("*.mp3"))
    assert not list(output_dir.rglob("*.wav"))
    assert not list(output_dir.rglob("*.flac"))
    manifest = json.loads((output_dir / "testing_stuff_music_corpus_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_started"] is False
    assert manifest["fine_tune_started"] is False
    assert manifest["audio_exported"] is False


def test_main_checks_docker_runtime_before_parsing_args(monkeypatch) -> None:
    calls = []

    def reject_non_docker_runtime(**kwargs):
        calls.append(("require_docker_runtime", kwargs))
        raise RuntimeError("Docker-only test sentinel")

    def fail_if_args_are_parsed(self, argv=None):
        raise AssertionError("parse_args must not run before Docker runtime validation")

    monkeypatch.setattr(corpus_plan, "require_docker_runtime", reject_non_docker_runtime)
    monkeypatch.setattr(corpus_plan.argparse.ArgumentParser, "parse_args", fail_if_args_are_parsed)

    try:
        corpus_plan.main(["--source-root", "/host_testing_stuff_music", "--output-dir", "/tmp/out"])
    except RuntimeError as exc:
        assert str(exc) == "Docker-only test sentinel"
    else:
        raise AssertionError("main must fail through Docker runtime guard")

    assert calls == [
        (
            "require_docker_runtime",
            {
                "tool_name": "build_testing_stuff_music_corpus_plan.py",
                "entrypoint_path": "/app/tools/build_testing_stuff_music_corpus_plan.py",
            },
        )
    ]


def test_direct_script_invocation_rejects_non_mounted_source_root(tmp_path: Path) -> None:
    source_root = tmp_path / "testing_stuff_music"
    touch_audio(source_root / "song.mp3")
    output_dir = tmp_path / "direct"

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_testing_stuff_music_corpus_plan.py",
            "--source-root",
            str(source_root),
            "--output-dir",
            str(output_dir),
            "--docker-root-label",
            "/host_testing_stuff_music",
            "--host-root-label",
            "D:/AIMusicProject/testing_stuff_music",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "source_root_must_be_docker_mount:/host_testing_stuff_music" in result.stderr
    assert not (output_dir / "testing_stuff_music_corpus_plan.jsonl").exists()
