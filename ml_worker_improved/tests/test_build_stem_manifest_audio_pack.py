from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_stem_manifest_audio_pack import build_stem_manifest_audio_pack


REQUIRED_FULL_PACK_STEMS = ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"]


def _write_audio(path: Path, *, amplitude: float = 0.05, duration_sec: float = 1.0) -> Path:
    sample_rate = 16_000
    frames = int(sample_rate * duration_sec)
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    mono = np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32) * amplitude
    stereo = np.column_stack([mono, mono])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), stereo, sample_rate, format="FLAC", subtype="PCM_24")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _source_manifest(tmp_path: Path, *, release_ready: bool = False, omit_stem: str = "") -> Path:
    stems = {
        "vocals": {"path": str(_write_audio(tmp_path / "src" / "vocals.flac"))},
        "minus": {"path": str(_write_audio(tmp_path / "src" / "minus.flac"))},
        "drums": {"path": str(_write_audio(tmp_path / "src" / "drums.flac"))},
        "bass": {"path": str(_write_audio(tmp_path / "src" / "bass.flac"))},
        "other": {"path": str(_write_audio(tmp_path / "src" / "other.flac"))},
        "guitar": {"path": str(_write_audio(tmp_path / "src" / "guitar.flac"))},
        "piano": {"path": str(_write_audio(tmp_path / "src" / "piano.flac"))},
    }
    if omit_stem:
        stems.pop(omit_stem)
    return _write_json(
        tmp_path / "stem_manifest.json",
        {
            "audio_path": "/app/testing_flstudio_music/Square a Saw - Oxygen.flac",
            "model_name": "htdemucs_6s",
            "target_stem": "minus",
            "delivery_allowed": False,
            "presentation_status": "diagnostic_candidate_not_for_listen_pack",
            "release_status": "not_release_ready",
            "release_ready": release_ready,
            "contract": {
                "registered_model_candidate": True,
                "application_reference_free": True,
                "fallback_used": False,
                "hidden_inference": False,
                "per_track_hardcode_allowed": False,
                "not_timestamp_hardcoded": True,
            },
            "execution_contract": {
                "gpu_inference_required": True,
                "gpu_inference_skipped": False,
                "cuda_preflight": {"cuda_available": True, "resolved_device": "cuda"},
            },
            "stems": stems,
        },
    )


def test_stem_manifest_audio_pack_copies_6s_stems_with_short_names(tmp_path: Path) -> None:
    track_id = "Square_a_Saw_-_Oxygen__Tevlo_Vocals_Resynthesized"
    manifest = build_stem_manifest_audio_pack(
        stem_manifest_json=_source_manifest(tmp_path),
        output_dir=tmp_path / "pack",
        track_id=track_id,
        stems=("vocal", "minus", "drums", "bass", "other", "guitar", "piano"),
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["release_ready"] is False
    assert manifest["stem_count"] == 7
    by_stem = {item["stem_type"]: Path(item["listen_audio_path"]) for item in manifest["stems"]}
    assert by_stem["vocal"].relative_to(tmp_path / "pack").as_posix() == "audio/t001/vocal.flac"
    assert by_stem["minus"].relative_to(tmp_path / "pack").as_posix() == "audio/t001/minus.flac"
    assert by_stem["drums"].name == "drums.flac"
    assert by_stem["guitar"].name == "guitar.flac"
    assert all(path.exists() for path in by_stem.values())
    assert all(track_id not in str(path) for path in by_stem.values())
    assert {item["track_alias"] for item in manifest["stems"]} == {"t001"}
    assert {item["source_track_id"] for item in manifest["stems"]} == {track_id}
    assert manifest["semantic_quality_verified"] is False
    assert manifest["can_publish_as_result"] is False


def test_stem_manifest_audio_pack_can_use_explicit_stem_audio_override(tmp_path: Path) -> None:
    source_manifest = _source_manifest(tmp_path)
    override_minus = _write_audio(tmp_path / "normalized" / "minus.flac", amplitude=0.11)

    manifest = build_stem_manifest_audio_pack(
        stem_manifest_json=source_manifest,
        output_dir=tmp_path / "pack",
        track_id="track-a",
        stems=("minus",),
        override_stem_paths={"minus": override_minus},
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["stem_count"] == 1
    row = manifest["stems"][0]
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    assert row["stem_type"] == "minus"
    assert row["source_audio_path"] == str(override_minus)
    assert row["source_manifest_audio_path"] == source_payload["stems"]["minus"]["path"]
    assert row["audio_override_applied"] is True
    assert manifest["source_audio_overrides"] == {"minus": str(override_minus)}
    assert Path(row["listen_audio_path"]).read_bytes() == override_minus.read_bytes()
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False


def test_stem_manifest_audio_pack_maps_mac_local_paths_to_artifact_mirror(tmp_path: Path) -> None:
    source_manifest = _source_manifest(tmp_path)
    mirror_dir = tmp_path / "artifacts" / "mac_api" / "altm002" / "bci"
    vocal_path = _write_audio(mirror_dir / "vocals.flac")
    minus_path = _write_audio(mirror_dir / "instrumental.flac", amplitude=0.08)
    mac_source_dir = "/Users/andrejviluha/aimusic_local_artifacts/mac_api/altm002/bci"
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    payload["stems"]["vocals"]["path"] = f"{mac_source_dir}/vocals.flac"
    payload["stems"]["minus"]["path"] = f"{mac_source_dir}/instrumental.flac"
    source_manifest.write_text(json.dumps(payload), encoding="utf-8")

    manifest = build_stem_manifest_audio_pack(
        stem_manifest_json=source_manifest,
        output_dir=tmp_path / "pack",
        track_id="track-a",
        stems=("vocal", "minus"),
        path_prefix_map=[(mac_source_dir, str(mirror_dir))],
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["stem_count"] == 2
    assert manifest["path_prefix_map"] == [{"from": mac_source_dir, "to": str(mirror_dir)}]
    stems = {row["stem_type"]: row for row in manifest["stems"]}
    assert stems["vocal"]["source_manifest_audio_path"] == f"{mac_source_dir}/vocals.flac"
    assert stems["vocal"]["source_audio_path"] == str(vocal_path)
    assert stems["vocal"]["source_audio_path_resolution"]["mapped"] is True
    assert stems["minus"]["source_manifest_audio_path"] == f"{mac_source_dir}/instrumental.flac"
    assert stems["minus"]["source_audio_path"] == str(minus_path)
    assert stems["minus"]["source_audio_path_resolution"]["resolved_path"] == str(minus_path)


def test_stem_manifest_audio_pack_uses_instrumental_for_legacy_minus_product_alias(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    vocal_path = _write_audio(source_dir / "vocals.flac")
    instrumental_path = _write_audio(source_dir / "instrumental.flac", amplitude=0.08)
    minus_path = _write_audio(source_dir / "minus.flac", amplitude=0.08)
    source_manifest = _write_json(
        tmp_path / "stem_manifest.json",
        {
            "audio_path": str(tmp_path / "testing_stuff_music" / "source.mp3"),
            "model_name": "becruily_instrumental_vocals",
            "target_stem": "instrumental",
            "target_product_stem": "minus",
            "target_audio_path": str(instrumental_path),
            "target_product_audio_path": str(minus_path),
            "target_model_output_audio_path": str(instrumental_path),
            "role_safe_stem_contract": {
                "source_manifest_target_stem": "instrumental",
                "canonical_stem_type": "instrumental",
                "user_facing_stem_role": "minus",
                "manifest_stem_name": "instrumental",
                "product_stem_type": "minus",
                "model_output_stem_type": "instrumental",
                "naming_contract": "source_manifest_uses_instrumental_stem; judge_rows_use_minus_product_role",
            },
            "target_stem_artifact": {
                "path": str(instrumental_path),
                "target_stem": "instrumental",
                "source_manifest_target_stem": "instrumental",
                "canonical_stem_type": "instrumental",
                "user_facing_stem_role": "minus",
                "manifest_stem_name": "instrumental",
                "product_stem_type": "minus",
                "model_output_stem_type": "instrumental",
                "product_audio_path": str(minus_path),
                "model_output_audio_path": str(instrumental_path),
                "naming_contract": "source_manifest_uses_instrumental_stem; judge_rows_use_minus_product_role",
            },
            "delivery_allowed": False,
            "presentation_status": "diagnostic_candidate_not_for_listen_pack",
            "release_status": "not_release_ready",
            "release_ready": False,
            "contract": {
                "registered_model_candidate": True,
                "application_reference_free": True,
                "fallback_used": False,
                "hidden_inference": False,
                "per_track_hardcode_allowed": False,
                "not_timestamp_hardcoded": True,
            },
            "execution_contract": {
                "gpu_inference_required": True,
                "gpu_inference_skipped": False,
                "cuda_preflight": {"cuda_available": True, "resolved_device": "cuda"},
            },
            "stems": {
                "vocals": {"path": str(vocal_path), "role": "companion"},
                "instrumental": {"path": str(instrumental_path), "role": "target"},
                "minus": {
                    "path": str(minus_path),
                    "role": "product_alias",
                    "alias_of": "instrumental",
                    "source_manifest_target_stem": "instrumental",
                    "canonical_stem_type": "instrumental",
                    "user_facing_stem_role": "minus",
                    "manifest_stem_name": "instrumental",
                    "product_stem_type": "minus",
                    "model_output_stem_type": "instrumental",
                    "product_audio_path": str(minus_path),
                    "model_output_audio_path": str(instrumental_path),
                    "naming_contract": "source_manifest_uses_instrumental_stem; judge_rows_use_minus_product_role",
                },
            },
        },
    )

    manifest = build_stem_manifest_audio_pack(
        stem_manifest_json=source_manifest,
        output_dir=tmp_path / "pack",
        track_id="track-a",
        stems=("minus",),
    )

    row = manifest["stems"][0]
    assert row["stem_type"] == "minus"
    assert row["source_stem_key"] == "instrumental"
    assert row["source_manifest_target_stem"] == "instrumental"
    assert row["canonical_stem_type"] == "instrumental"
    assert row["user_facing_stem_role"] == "minus"
    assert row["manifest_stem_name"] == "instrumental"
    assert row["product_stem_type"] == "minus"
    assert row["model_output_stem_type"] == "instrumental"
    assert row["source_manifest_stem_type"] == "instrumental"
    assert row["naming_contract"] == "source_manifest_uses_instrumental_stem; judge_rows_use_minus_product_role"
    assert row["product_audio_path"] == str(instrumental_path)
    assert row["model_output_audio_path"] == str(instrumental_path)
    assert row["source_manifest_audio_path"] == str(instrumental_path)
    assert row["source_audio_path"] == str(instrumental_path)
    assert Path(row["listen_audio_path"]).name == "minus.flac"


def test_stem_manifest_audio_pack_blocks_release_ready_source(tmp_path: Path) -> None:
    manifest = build_stem_manifest_audio_pack(
        stem_manifest_json=_source_manifest(tmp_path, release_ready=True),
        output_dir=tmp_path / "pack",
        track_id="track-a",
        stems=("drums",),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["stem_count"] == 0
    assert "source_release_ready_true" in manifest["failed_checks"]
    assert not (tmp_path / "pack" / "audio").exists()


def test_stem_manifest_audio_pack_blocks_missing_required_stem(tmp_path: Path) -> None:
    manifest = build_stem_manifest_audio_pack(
        stem_manifest_json=_source_manifest(tmp_path, omit_stem="piano"),
        output_dir=tmp_path / "pack",
        track_id="track-a",
        stems=("piano",),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["stem_count"] == 0
    assert "source_stem_missing:piano" in manifest["failed_checks"]
    assert not (tmp_path / "pack" / "audio").exists()


def test_stem_manifest_audio_pack_full_pack_contract_persists_missing_stems(tmp_path: Path) -> None:
    manifest = build_stem_manifest_audio_pack(
        stem_manifest_json=_source_manifest(tmp_path, omit_stem="piano"),
        output_dir=tmp_path / "pack",
        track_id="track-a",
        require_full_pack_stems=True,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["stem_count"] == 0
    assert manifest["required_stems"] == REQUIRED_FULL_PACK_STEMS
    assert manifest["present_stems"] == ["vocal", "minus", "drums", "bass", "other", "guitar"]
    assert manifest["missing_required_stems"] == ["piano"]
    assert "source_stem_missing:piano" in manifest["failed_checks"]
    assert "required_stem_missing:piano" in manifest["failed_checks"]
    assert not (tmp_path / "pack" / "audio").exists()
