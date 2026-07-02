from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.validate_aggregate_short_listen_pack import validate_aggregate_short_listen_pack


REQUIRED_FULL_PACK_STEMS = ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"]


def _write_audio(path: Path) -> Path:
    sample_rate = 16_000
    t = np.arange(sample_rate // 4, dtype=np.float32) / float(sample_rate)
    mono = (np.sin(2.0 * np.pi * 220.0 * t) * 0.05).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), stereo, sample_rate, format="FLAC", subtype="PCM_24")
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


def _valid_pack(tmp_path: Path) -> Path:
    audio = _write_audio(tmp_path / "pack" / "audio" / "voc" / "v001.flac")
    selected = [
        {
            "schema": "aggregate_short_listen_pack_audio.v1",
            "source_alias": "voc",
            "source_manifest": str(tmp_path / "source" / "pack.json"),
            "source_audio_path": str(audio),
            "audio_path": "audio/voc/v001.flac",
            "stem_type": "vocal",
            "track_alias": "t001",
            "quality_status": "listen_preview_only",
            "semantic_quality_verified": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "sample_rate": 16_000,
            "channels": 2,
            "duration_sec": 0.25,
            "sha256": _sha256(audio),
        }
    ]
    selected_path = tmp_path / "pack" / "selected.jsonl"
    selected_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in selected) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "pack" / "progress.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "start", "pack_id": "lp_test", "sources": [{"alias": "voc"}]}, sort_keys=True),
                json.dumps(
                    {
                        "event": "finish",
                        "delivery_status": "aggregate_validation_pending",
                        "item_count": 1,
                        "failed_checks": [],
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "pack" / "README.md").write_text(
        "# lp_test\n\n- `pack.json`\n- `coverage.json`\n- `selected.jsonl`\n- `progress.jsonl`\n- `audio/voc/v001.flac`\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "pack" / "coverage.json",
        {
            "schema": "aggregate_short_listen_pack_coverage.v1",
            "pack_id": "lp_test",
            "release_ready": False,
            "can_publish_as_result": False,
            "stem_groups": [
                {
                    "group_alias": "voc",
                    "status": "listen_preview_only",
                    "item_count": 1,
                    "stem_types": ["vocal"],
                    "files": ["audio/voc/v001.flac"],
                    "semantic_quality_verified": False,
                }
            ],
            "failed_checks": [],
            "safety": {"training_started": False, "thresholds_relaxed": False},
        },
    )
    _write_json(
        tmp_path / "pack" / "pack.json",
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "pack_id": "lp_test",
            "delivery_status": "aggregate_validation_pending",
            "delivery_allowed": False,
            "presentation_status": "aggregate_validation_pending_not_for_listen_pack",
            "release_status": "aggregate_validation_pending",
            "release_ready": False,
            "can_publish_as_result": False,
            "listenable": False,
            "deliverable": False,
            "aggregate_validation_ready": True,
            "item_count": 1,
            "items": selected,
            "failed_checks": [],
            "coverage_json": str(tmp_path / "pack" / "coverage.json"),
            "selected_jsonl": str(selected_path),
            "progress_jsonl": str(tmp_path / "pack" / "progress.jsonl"),
            "artifact_name_policy": {
                "schema": "short_windows_safe_artifact_names.v1",
                "audio_dir": "audio",
                "source_alias_strategy": "caller_supplied_validated_short_alias",
                "raw_track_or_stem_ids_in_path": False,
                "provenance_in_manifest": True,
                "metadata_filenames": ["pack.json", "coverage.json", "selected.jsonl", "progress.jsonl", "README.md"],
            },
            "contract": {
                "training_started": False,
                "separator_training_started": False,
                "models_changed": False,
                "model_registry_edited": False,
                "thresholds_relaxed": False,
                "cleanup_routes_added": False,
                "release_ready_forced_false": True,
            },
        },
    )
    return tmp_path / "pack" / "pack.json"


def test_validate_aggregate_short_listen_pack_accepts_valid_pack(tmp_path: Path) -> None:
    manifest = validate_aggregate_short_listen_pack(
        pack_manifest_json=_valid_pack(tmp_path),
        output_dir=tmp_path / "validation",
    )

    assert manifest["validation_status"] == "passed"
    assert manifest["failed_checks"] == []
    assert manifest["audio_file_count"] == 1
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False


def test_validate_aggregate_short_listen_pack_full_pack_contract_blocks_missing_stems(tmp_path: Path) -> None:
    manifest = validate_aggregate_short_listen_pack(
        pack_manifest_json=_valid_pack(tmp_path),
        output_dir=tmp_path / "validation",
        required_stems=REQUIRED_FULL_PACK_STEMS,
    )

    assert manifest["validation_status"] == "failed"
    assert manifest["audio_file_count"] == 0
    assert manifest["required_stems"] == REQUIRED_FULL_PACK_STEMS
    assert manifest["present_stems"] == ["vocal"]
    assert manifest["missing_required_stems"] == ["minus", "drums", "bass", "other", "guitar", "piano"]
    assert "required_stem_missing:minus" in manifest["failed_checks"]


def test_validate_aggregate_short_listen_pack_reads_required_stems_from_pack_manifest(tmp_path: Path) -> None:
    pack = _valid_pack(tmp_path)
    payload = json.loads(pack.read_text(encoding="utf-8"))
    payload["required_stems"] = REQUIRED_FULL_PACK_STEMS
    payload["present_stems"] = ["vocal"]
    payload["missing_required_stems"] = ["minus", "drums", "bass", "other", "guitar", "piano"]
    pack.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    manifest = validate_aggregate_short_listen_pack(
        pack_manifest_json=pack,
        output_dir=tmp_path / "validation",
    )

    assert manifest["validation_status"] == "failed"
    assert manifest["required_stems"] == REQUIRED_FULL_PACK_STEMS
    assert manifest["present_stems"] == ["vocal"]
    assert manifest["missing_required_stems"] == ["minus", "drums", "bass", "other", "guitar", "piano"]
    assert "required_stem_missing:minus" in manifest["failed_checks"]


def test_validate_aggregate_short_listen_pack_blocks_required_unverified_stem(tmp_path: Path) -> None:
    pack = _valid_pack(tmp_path)
    payload = json.loads(pack.read_text(encoding="utf-8"))
    payload["required_stems"] = ["drums"]
    payload["items"][0]["stem_type"] = "drums"
    payload["items"][0]["semantic_quality_verified"] = False
    payload["items"][0]["quality_status"] = "listen_preview_only"
    payload["present_stems"] = ["drums"]
    payload["missing_required_stems"] = []
    pack.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    selected_path = tmp_path / "pack" / "selected.jsonl"
    selected_path.write_text(json.dumps(payload["items"][0], sort_keys=True) + "\n", encoding="utf-8")
    coverage_path = tmp_path / "pack" / "coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["stem_groups"][0]["stem_types"] = ["drums"]
    coverage["stem_groups"][0]["semantic_quality_verified"] = False
    coverage_path.write_text(json.dumps(coverage, sort_keys=True), encoding="utf-8")

    manifest = validate_aggregate_short_listen_pack(
        pack_manifest_json=pack,
        output_dir=tmp_path / "validation",
    )

    assert manifest["validation_status"] == "failed"
    assert "required_stem_quality_not_verified:drums" in manifest["failed_checks"]


def test_validate_aggregate_short_listen_pack_blocks_mixed_required_unverified_stem_rows(tmp_path: Path) -> None:
    pack = _valid_pack(tmp_path)
    payload = json.loads(pack.read_text(encoding="utf-8"))
    second_audio = _write_audio(tmp_path / "pack" / "audio" / "voc" / "v002.flac")
    first_item = dict(payload["items"][0])
    first_item["stem_type"] = "drums"
    first_item["semantic_quality_verified"] = True
    first_item["quality_status"] = "listen_preview_only"
    second_item = dict(first_item)
    second_item["audio_path"] = "audio/voc/v002.flac"
    second_item["source_audio_path"] = str(second_audio)
    second_item["semantic_quality_verified"] = False
    second_item["sha256"] = _sha256(second_audio)
    payload["required_stems"] = ["drums"]
    payload["items"] = [first_item, second_item]
    payload["item_count"] = 2
    payload["present_stems"] = ["drums"]
    payload["missing_required_stems"] = []
    pack.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    selected_path = tmp_path / "pack" / "selected.jsonl"
    selected_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in payload["items"]) + "\n",
        encoding="utf-8",
    )
    coverage_path = tmp_path / "pack" / "coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["stem_groups"][0]["item_count"] = 2
    coverage["stem_groups"][0]["stem_types"] = ["drums"]
    coverage["stem_groups"][0]["files"] = ["audio/voc/v001.flac", "audio/voc/v002.flac"]
    coverage["stem_groups"][0]["semantic_quality_verified"] = True
    coverage_path.write_text(json.dumps(coverage, sort_keys=True), encoding="utf-8")
    (tmp_path / "pack" / "README.md").write_text(
        "# lp_test\n\n- `pack.json`\n- `coverage.json`\n- `selected.jsonl`\n- `progress.jsonl`\n- `audio/voc/v001.flac`\n- `audio/voc/v002.flac`\n",
        encoding="utf-8",
    )

    manifest = validate_aggregate_short_listen_pack(
        pack_manifest_json=pack,
        output_dir=tmp_path / "validation",
    )

    assert manifest["validation_status"] == "failed"
    assert "required_stem_quality_not_verified:drums" in manifest["failed_checks"]


def test_validate_aggregate_short_listen_pack_blocks_long_output_segment(tmp_path: Path) -> None:
    pack = _valid_pack(tmp_path)
    payload = json.loads(pack.read_text(encoding="utf-8"))
    payload["items"][0]["audio_path"] = "audio/voc/this_filename_is_far_too_long_for_windows.flac"
    pack.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    manifest = validate_aggregate_short_listen_pack(
        pack_manifest_json=pack,
        output_dir=tmp_path / "validation",
    )

    assert manifest["validation_status"] == "failed"
    assert "long_path_segment" in manifest["failed_checks"]


def test_validate_aggregate_short_listen_pack_blocks_release_ready(tmp_path: Path) -> None:
    pack = _valid_pack(tmp_path)
    payload = json.loads(pack.read_text(encoding="utf-8"))
    payload["release_ready"] = True
    pack.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    manifest = validate_aggregate_short_listen_pack(
        pack_manifest_json=pack,
        output_dir=tmp_path / "validation",
    )

    assert manifest["validation_status"] == "failed"
    assert "pack_release_ready_true" in manifest["failed_checks"]


def test_validate_aggregate_short_listen_pack_blocks_missing_audio(tmp_path: Path) -> None:
    pack = _valid_pack(tmp_path)
    payload = json.loads(pack.read_text(encoding="utf-8"))
    payload["items"][0]["audio_path"] = "audio/voc/missing.flac"
    pack.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    manifest = validate_aggregate_short_listen_pack(
        pack_manifest_json=pack,
        output_dir=tmp_path / "validation",
    )

    assert manifest["validation_status"] == "failed"
    assert "audio_missing" in manifest["failed_checks"]


def test_validate_aggregate_short_listen_pack_blocks_audio_sha256_mismatch(tmp_path: Path) -> None:
    pack = _valid_pack(tmp_path)
    (tmp_path / "pack" / "audio" / "voc" / "v001.flac").write_bytes(b"not the original audio")

    manifest = validate_aggregate_short_listen_pack(
        pack_manifest_json=pack,
        output_dir=tmp_path / "validation",
    )

    assert manifest["validation_status"] == "failed"
    assert "audio_sha256_mismatch" in manifest["failed_checks"]


def test_validate_aggregate_short_listen_pack_blocks_manifest_failed_checks(tmp_path: Path) -> None:
    pack = _valid_pack(tmp_path)
    payload = json.loads(pack.read_text(encoding="utf-8"))
    payload["failed_checks"] = ["latent_failure"]
    pack.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    manifest = validate_aggregate_short_listen_pack(
        pack_manifest_json=pack,
        output_dir=tmp_path / "validation",
    )

    assert manifest["validation_status"] == "failed"
    assert "pack_failed_checks_not_empty" in manifest["failed_checks"]


def test_validate_aggregate_short_listen_pack_requires_promised_artifacts(tmp_path: Path) -> None:
    pack = _valid_pack(tmp_path)
    (tmp_path / "pack" / "selected.jsonl").unlink()

    manifest = validate_aggregate_short_listen_pack(
        pack_manifest_json=pack,
        output_dir=tmp_path / "validation",
    )

    assert manifest["validation_status"] == "failed"
    assert "selected_jsonl_missing" in manifest["failed_checks"]
