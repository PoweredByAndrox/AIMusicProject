from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "collect_no_reference_minus_candidate_pool.py"


def load_module():
    spec = importlib.util.spec_from_file_location("collect_no_reference_minus_candidate_pool", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def touch_audio(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not real audio; collector must not read contents")
    return str(path)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_collects_existing_minus_candidates_without_vocal_pair(tmp_path: Path) -> None:
    module = load_module()
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    minus_a = touch_audio(tmp_path / "track-a" / "model-a" / "instrumental.flac")
    minus_b = touch_audio(tmp_path / "track-a" / "model-b" / "minus.flac")
    write_json(
        tmp_path / "track-a" / "model-a" / "stem_manifest.json",
        {
            "source_audio_path": mix_path,
            "candidate_id": "model-a",
            "artifacts": {"instrumental_flac": minus_a},
            "delivery_status": "diagnostic",
        },
    )
    write_json(
        tmp_path / "track-a" / "model-b" / "stem_manifest.json",
        {
            "source_audio_path": mix_path,
            "candidate_id": "model-b",
            "selected_minus_path": minus_b,
        },
    )

    result = module.collect_minus_candidate_pool(tmp_path)

    assert result["summary"]["complete_minus_candidates"] == 2
    assert result["summary"]["tracks_with_2plus_minus_candidates"] == 1
    assert [row["minus_path"] for row in result["rows"]] == [minus_a, minus_b]
    assert all(row["instrumental_path"] == row["minus_path"] for row in result["rows"])
    assert all(row["is_complete_pair"] is True for row in result["rows"])
    assert all(row["vocal_path"] == "" for row in result["rows"])
    assert result["summary"]["safety"]["audio_read"] is False
    assert result["summary"]["safety"]["audio_exported"] is False
    assert result["summary"]["safety"]["models_changed"] is False


def test_skips_stem_only_manifest_without_rendering_audio(tmp_path: Path) -> None:
    module = load_module()
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    bass = touch_audio(tmp_path / "track-a" / "htdemucs" / "bass.flac")
    drums = touch_audio(tmp_path / "track-a" / "htdemucs" / "drums.flac")
    vocals = touch_audio(tmp_path / "track-a" / "htdemucs" / "vocals.flac")
    write_json(
        tmp_path / "track-a" / "htdemucs" / "stem_manifest.json",
        {
            "source_audio_path": mix_path,
            "candidate_id": "htdemucs",
            "stems": {
                "bass": {"path": bass},
                "drums": {"path": drums},
                "vocals": {"path": vocals},
            },
        },
    )

    result = module.collect_minus_candidate_pool(tmp_path)

    assert result["rows"] == []
    assert result["summary"]["skipped_incomplete"] == 1
    assert result["skipped"][0]["skip_reason"] == "stem_only_requires_render_route"
    assert not list(tmp_path.rglob("instrumental_from_non_vocals.flac"))


def test_deduplicates_seed_and_manifest_minus_paths_deterministically(tmp_path: Path) -> None:
    module = load_module()
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    minus_path = touch_audio(tmp_path / "track-a" / "model-a" / "instrumental.flac")
    write_json(
        tmp_path / "track-a" / "model-a" / "stem_manifest.json",
        {
            "source_audio_path": mix_path,
            "candidate_id": "manifest-model-a",
            "artifacts": {"instrumental_flac": minus_path},
        },
    )
    seed_pool = write_jsonl(
        tmp_path / "seed_pool.jsonl",
        [
            {
                "track_id": "track-a",
                "candidate_id": "seed-model-a",
                "mix_path": mix_path,
                "minus_path": minus_path,
                "instrumental_path": minus_path,
                "vocal_path": "/ignored/vocal.flac",
                "is_complete_pair": True,
            }
        ],
    )

    first = module.collect_minus_candidate_pool(tmp_path, seed_candidate_pool_jsonls=[seed_pool])
    second = module.collect_minus_candidate_pool(tmp_path, seed_candidate_pool_jsonls=[seed_pool])

    assert first == second
    assert first["summary"]["complete_minus_candidates"] == 1
    assert first["summary"]["seed_minus_candidates"] == 1
    assert first["summary"]["deduplicated_minus_candidates"] == 1
    assert first["rows"][0]["minus_path"] == minus_path


def test_manifest_minus_row_preserves_paired_vocal_path(tmp_path: Path) -> None:
    module = load_module()
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    minus_path = touch_audio(tmp_path / "track-a" / "model-a" / "instrumental.flac")
    vocal_path = touch_audio(tmp_path / "track-a" / "model-a" / "vocals.flac")
    write_json(
        tmp_path / "track-a" / "model-a" / "stem_manifest.json",
        {
            "source_audio_path": mix_path,
            "candidate_id": "manifest-model-a",
            "artifacts": {"instrumental_flac": minus_path},
            "stems": {"vocals": {"path": vocal_path}},
        },
    )

    result = module.collect_minus_candidate_pool(tmp_path)

    assert result["summary"]["complete_minus_candidates"] == 1
    assert len(result["rows"]) == 1
    assert result["rows"][0]["minus_path"] == minus_path
    assert result["rows"][0]["vocal_path"] == vocal_path


def test_manifest_minus_row_uses_sibling_vocals_when_manifest_lacks_vocal_path(tmp_path: Path) -> None:
    module = load_module()
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    minus_path = touch_audio(tmp_path / "track-a" / "model-a" / "instrumental.flac")
    vocal_path = touch_audio(tmp_path / "track-a" / "model-a" / "vocals.flac")
    write_json(
        tmp_path / "track-a" / "model-a" / "stem_manifest.json",
        {
            "source_audio_path": mix_path,
            "candidate_id": "manifest-model-a",
            "artifacts": {"instrumental_flac": minus_path},
        },
    )

    result = module.collect_minus_candidate_pool(tmp_path)

    assert len(result["rows"]) == 1
    assert result["rows"][0]["minus_path"] == minus_path
    assert result["rows"][0]["vocal_path"] == vocal_path


def test_write_outputs_does_not_create_audio_files(tmp_path: Path) -> None:
    module = load_module()
    mix_path = touch_audio(tmp_path / "track-a" / "source.mp3")
    minus_path = touch_audio(tmp_path / "track-a" / "model-a" / "instrumental.flac")
    write_json(
        tmp_path / "track-a" / "model-a" / "stem_manifest.json",
        {
            "source_audio_path": mix_path,
            "candidate_id": "model-a",
            "artifacts": {"instrumental_flac": minus_path},
        },
    )
    result = module.collect_minus_candidate_pool(tmp_path)
    output_jsonl = tmp_path / "out" / "minus_pool.jsonl"
    summary_md = tmp_path / "out" / "minus_pool_summary.md"
    skipped_jsonl = tmp_path / "out" / "skipped.jsonl"

    module.write_outputs(result, output_jsonl, summary_md, skipped_jsonl)

    output_rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line]
    skipped_rows = [json.loads(line) for line in skipped_jsonl.read_text(encoding="utf-8").splitlines() if line]
    assert len(output_rows) == result["summary"]["complete_minus_candidates"]
    assert len(skipped_rows) == result["summary"]["skipped_incomplete"]
    assert summary_md.read_text(encoding="utf-8").startswith("# No-Reference Minus Candidate Pool Summary")
    assert not list((tmp_path / "out").rglob("*.wav"))
    assert not list((tmp_path / "out").rglob("*.flac"))
