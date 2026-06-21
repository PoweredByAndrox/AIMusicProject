from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_short_listen_replacement_pack import build_short_listen_replacement_pack


STEMS = ("vocal", "minus", "drums", "bass", "other", "guitar", "piano")


def _write_audio(path: Path, *, freq: float = 220.0) -> Path:
    sample_rate = 16_000
    t = np.arange(sample_rate // 4, dtype=np.float32) / float(sample_rate)
    mono = (np.sin(2.0 * np.pi * freq * t) * 0.05).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.column_stack([mono, mono]), sample_rate, format="FLAC", subtype="PCM_24")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _base_pack(tmp_path: Path) -> Path:
    items = []
    for index, stem in enumerate(STEMS, start=1):
        audio = _write_audio(tmp_path / "base" / "audio" / "s004" / "t001" / f"{stem}.flac", freq=180.0 + index)
        items.append(
            {
                "schema": "aggregate_short_listen_pack_audio.v1",
                "source_alias": "s004",
                "track_alias": "t001",
                "source_manifest": str(tmp_path / "source" / "s004.json"),
                "source_audio_path": str(audio),
                "audio_path": f"audio/s004/t001/{stem}.flac",
                "stem_type": stem,
                "quality_status": "listen_preview_only",
                "semantic_quality_verified": True,
                "release_ready": False,
                "can_publish_as_result": False,
                "sample_rate": 16_000,
                "channels": 2,
                "duration_sec": 0.25,
                "sha256": _sha256(audio),
                "source_row": {"stem_type": stem, "source_audio_path": str(audio)},
            }
        )
    selected = tmp_path / "base" / "selected.jsonl"
    selected.write_text("\n".join(json.dumps(row, sort_keys=True) for row in items) + "\n", encoding="utf-8")
    coverage = {
        "schema": "aggregate_short_listen_pack_coverage.v1",
        "pack_id": "base",
        "release_ready": False,
        "can_publish_as_result": False,
        "stem_groups": [
            {
                "group_alias": "s004",
                "status": "listen_preview_only",
                "item_count": len(items),
                "stem_types": list(STEMS),
                "files": [row["audio_path"] for row in items],
                "semantic_quality_verified": True,
            }
        ],
        "failed_checks": [],
    }
    _write_json(tmp_path / "base" / "coverage.json", coverage)
    (tmp_path / "base" / "progress.jsonl").write_text(
        json.dumps({"event": "start"}, sort_keys=True)
        + "\n"
        + json.dumps({"event": "finish", "delivery_status": "accepted", "item_count": len(items), "failed_checks": []}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "base" / "README.md").write_text(
        "# base\n\n- `pack.json`\n- `coverage.json`\n- `selected.jsonl`\n- `progress.jsonl`\n"
        + "\n".join(f"- `{row['audio_path']}`" for row in items)
        + "\n",
        encoding="utf-8",
    )
    return _write_json(
        tmp_path / "base" / "pack.json",
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "pack_id": "base",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "release_ready": False,
            "can_publish_as_result": False,
            "listenable": True,
            "deliverable": False,
            "item_count": len(items),
            "items": items,
            "failed_checks": [],
            "coverage_json": str(tmp_path / "base" / "coverage.json"),
            "selected_jsonl": str(selected),
            "progress_jsonl": str(tmp_path / "base" / "progress.jsonl"),
            "artifact_name_policy": {
                "schema": "short_windows_safe_artifact_names.v1",
                "audio_dir": "audio",
                "source_alias_strategy": "caller_supplied_validated_short_alias",
                "raw_track_or_stem_ids_in_path": False,
                "provenance_in_manifest": True,
                "metadata_filenames": ["pack.json", "coverage.json", "selected.jsonl", "progress.jsonl", "README.md"],
            },
            "contract": {"release_ready_forced_false": True},
        },
    )


def _replacement_pack(tmp_path: Path, source_minus_path: Path) -> Path:
    repaired = _write_audio(tmp_path / "mps" / "audio" / "min" / "m001.flac", freq=333.0)
    return _write_json(
        tmp_path / "mps" / "pack.json",
        {
            "schema": "short_minus_phase_repair_selection_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "listen_audio_files": [
                {
                    "schema": "short_minus_phase_repair_selection_audio.v1",
                    "artifact_relative_path": "audio/min/m001.flac",
                    "listen_audio_path": str(repaired),
                    "source_audio_path": str(repaired),
                    "source_candidate_id": "q001_source_minus__minus_side_attenuate_50",
                    "stem_type": "minus",
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "presentation_status": "accepted_for_listen_pack",
                    "release_ready": False,
                    "can_publish_as_result": False,
                    "source_phase_repair_manifest": {
                        "source_minus_path": str(source_minus_path),
                        "source_mix_path": str(tmp_path / "mix.flac"),
                        "source_vocal_path": str(tmp_path / "vocal.flac"),
                    },
                }
            ],
        },
    )


def _track_replacement_pack(tmp_path: Path) -> Path:
    repaired = _write_audio(tmp_path / "ft" / "audio" / "t001" / "minus.flac", freq=377.0)
    return _write_json(
        tmp_path / "ft" / "pack.json",
        {
            "schema": "stem_manifest_audio_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "stems": [
                {
                    "schema": "stem_manifest_audio_pack_stem.v1",
                    "artifact_relative_path": "audio/t001/minus.flac",
                    "listen_audio_path": str(repaired),
                    "source_audio_path": str(repaired),
                    "source_track_id": "trk005",
                    "track_id": "trk005",
                    "stem_type": "minus",
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "presentation_status": "accepted_for_listen_pack",
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            ],
        },
    )


def _nested_source_replacement_pack(tmp_path: Path, source_audio_path: Path) -> Path:
    repaired = _write_audio(tmp_path / "nested" / "audio" / "min" / "m001.flac", freq=411.0)
    return _write_json(
        tmp_path / "nested" / "pack.json",
        {
            "schema": "short_listen_repair_selection_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "listen_audio_files": [
                {
                    "schema": "short_listen_repair_selection_audio.v1",
                    "artifact_relative_path": "audio/min/m001.flac",
                    "listen_audio_path": str(repaired),
                    "source_audio_path": str(repaired),
                    "source_candidate_id": "q002_minus_side_attenuate_50",
                    "stem_type": "minus",
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "presentation_status": "accepted_for_listen_pack",
                    "release_ready": False,
                    "can_publish_as_result": False,
                    "source_row": {
                        "source_audio_path": str(source_audio_path),
                        "track_id": "trk_shared",
                    },
                }
            ],
        },
    )


def test_replacement_pack_preserves_complete_group_and_short_paths(tmp_path: Path) -> None:
    base_pack = _base_pack(tmp_path)
    base_payload = json.loads(base_pack.read_text(encoding="utf-8"))
    source_minus_path = Path(next(row for row in base_payload["items"] if row["stem_type"] == "minus")["source_audio_path"])
    replacement_pack = _replacement_pack(tmp_path, source_minus_path)

    manifest = build_short_listen_replacement_pack(
        base_pack_json=base_pack,
        replacement_pack_jsons=[replacement_pack],
        output_dir=tmp_path / "out",
        pack_id="lp_test_replaced",
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["item_count"] == len(STEMS)
    assert manifest["replacement_count"] == 1
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    minus = next(row for row in manifest["items"] if row["stem_type"] == "minus")
    assert minus["source_alias"] == "s004"
    assert minus["track_alias"] == "t001"
    assert minus["audio_path"] == "audio/s004/t001/minus.flac"
    assert minus["replacement_applied"] is True
    assert minus["source_row"]["source_candidate_id"] == "q001_source_minus__minus_side_attenuate_50"
    assert "q001_source_minus" not in minus["audio_path"]
    assert (tmp_path / "out" / minus["audio_path"]).read_bytes() == Path(
        minus["replacement_source_audio_path"]
    ).read_bytes()

    coverage = json.loads((tmp_path / "out" / "coverage.json").read_text(encoding="utf-8"))
    groups = coverage["stem_groups"]
    assert len(groups) == 1
    assert groups[0]["group_alias"] == "s004"
    assert set(groups[0]["stem_types"]) == set(STEMS)
    assert groups[0]["item_count"] == len(STEMS)


def test_replacement_pack_can_match_direct_minus_by_track_id(tmp_path: Path) -> None:
    base_pack = _base_pack(tmp_path)
    base_payload = json.loads(base_pack.read_text(encoding="utf-8"))
    for row in base_payload["items"]:
        row["track_id"] = "trk005"
        row["source_row"]["track_id"] = "trk005"
        row["source_row"]["source_track_id"] = "trk005"
    base_pack.write_text(json.dumps(base_payload, sort_keys=True), encoding="utf-8")

    manifest = build_short_listen_replacement_pack(
        base_pack_json=base_pack,
        replacement_pack_jsons=[_track_replacement_pack(tmp_path)],
        output_dir=tmp_path / "out",
        pack_id="lp_test_replaced",
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["replacement_count"] == 1
    minus = next(row for row in manifest["items"] if row["stem_type"] == "minus")
    assert minus["audio_path"] == "audio/s004/t001/minus.flac"
    assert minus["replacement_match_key"] == "track:trk005:minus"
    assert minus["source_row"]["schema"] == "stem_manifest_audio_pack_stem.v1"


def test_replacement_pack_prefers_nested_source_path_over_shared_track_id(tmp_path: Path) -> None:
    base_pack = _base_pack(tmp_path)
    base_payload = json.loads(base_pack.read_text(encoding="utf-8"))
    minus = next(row for row in base_payload["items"] if row["stem_type"] == "minus")
    other = next(row for row in base_payload["items"] if row["stem_type"] == "other")
    other["stem_type"] = "minus"
    other["audio_path"] = "audio/s004/t001/minus_alt.flac"
    other["source_row"] = {"source_audio_path": other["source_audio_path"], "track_id": "trk_shared"}
    minus["source_row"] = {"source_audio_path": minus["source_audio_path"], "track_id": "trk_shared"}
    base_pack.write_text(json.dumps(base_payload, sort_keys=True), encoding="utf-8")

    manifest = build_short_listen_replacement_pack(
        base_pack_json=base_pack,
        replacement_pack_jsons=[_nested_source_replacement_pack(tmp_path, Path(minus["source_audio_path"]))],
        output_dir=tmp_path / "out",
        pack_id="lp_test_replaced",
    )

    assert manifest["delivery_status"] == "accepted"
    replaced = [row for row in manifest["items"] if row["stem_type"] == "minus" and row["replacement_applied"]]
    assert len(replaced) == 1
    assert replaced[0]["audio_path"] == "audio/s004/t001/minus.flac"
    assert replaced[0]["replacement_match_key"] == str(Path(minus["source_audio_path"]).resolve())
