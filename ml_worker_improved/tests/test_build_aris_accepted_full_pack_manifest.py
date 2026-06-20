from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_aris_accepted_full_pack_manifest import build_aris_accepted_full_pack_manifest


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


def _write_audio(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"audio")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "artifact_rag"
    audio = root / "src"
    vocal = _write_audio(audio / "vocal.flac")
    minus = _write_audio(audio / "minus.flac")
    sixs_paths = {stem: _write_audio(audio / f"{stem}.flac") for stem in ("drums", "bass", "other", "guitar", "piano")}
    accepted = _write_jsonl(
        root / "runs" / "r009" / "accepted.jsonl",
        [
            {
                "schema": "aris_refinement_acceptance_row.v1",
                "track_alias": "t001",
                "delivery_status": "accepted",
                "delivery_allowed": True,
                "presentation_status": "accepted_for_listen_pack",
                "objective_evidence": {
                    "reference_gate": "not_available",
                    "no_reference_gate": "passed",
                    "pair_consistency_gate": "passed",
                },
            }
        ],
    )
    candidates = _write_jsonl(
        root / "runs" / "r009" / "candidates.jsonl",
        [
            {"track_alias": "t001", "track_id": "track-1", "stem": "vocal", "target_audio_path": str(vocal)},
            {"track_alias": "t001", "track_id": "track-1", "stem": "minus", "target_audio_path": str(minus)},
        ],
    )
    sixs_manifest = _write_json(
        root / "6s" / "r001" / "stem_manifest.json",
        {
            "audio_path": "/source/song.mp3",
            "model_name": "htdemucs_6s",
            "contract": {
                "registered_model_candidate": True,
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
            "stems": {stem: {"path": str(path)} for stem, path in sixs_paths.items()},
            "delivery_allowed": False,
            "release_ready": False,
        },
    )
    return root, accepted, candidates, sixs_manifest


def test_builds_pack_source_manifest_from_accepted_pair_and_6s_stems(tmp_path: Path) -> None:
    root, accepted, candidates, sixs_manifest = _inputs(tmp_path)

    manifest = build_aris_accepted_full_pack_manifest(
        artifact_rag_root=root,
        accepted_jsonl=accepted,
        candidates_jsonl=candidates,
        sixs_stem_manifest_json=sixs_manifest,
        output_dir=root / "packsrc" / "r001",
    )

    assert manifest["status"] == "ready_for_pack_build"
    assert manifest["delivery_allowed"] is False
    source = _read_json(Path(manifest["source_manifest_json"]))
    assert source["schema"] == "aris_accepted_full_pack_source_manifest.v1"
    assert source["contract"]["registered_model_candidate"] is True
    assert source["delivery_allowed"] is False
    assert set(source["stems"]) == {"vocal", "minus", "drums", "bass", "other", "guitar", "piano"}
    assert source["stems"]["vocal"]["source"] == "accepted_vocal_minus_matrix"
    assert source["stems"]["drums"]["source"] == "htdemucs_6s"
    assert source["objective_evidence"]["no_reference_gate"] == "passed"


def test_rejects_output_outside_artifact_rag(tmp_path: Path) -> None:
    root, accepted, candidates, sixs_manifest = _inputs(tmp_path)

    with pytest.raises(ValueError, match="output_dir must be inside artifact_rag_root"):
        build_aris_accepted_full_pack_manifest(
            artifact_rag_root=root,
            accepted_jsonl=accepted,
            candidates_jsonl=candidates,
            sixs_stem_manifest_json=sixs_manifest,
            output_dir=tmp_path / "outside",
        )


def test_rejects_stem_audio_outside_artifact_rag(tmp_path: Path) -> None:
    root, accepted, candidates, sixs_manifest = _inputs(tmp_path)
    outside_vocal = _write_audio(tmp_path / "outside_audio" / "vocal.flac")
    candidates.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "track_alias": "t001",
                        "track_id": "track-1",
                        "stem": "vocal",
                        "target_audio_path": str(outside_vocal),
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "track_alias": "t001",
                        "track_id": "track-1",
                        "stem": "minus",
                        "target_audio_path": str(root / "src" / "minus.flac"),
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stem audio path must be inside artifact_rag_root: vocal"):
        build_aris_accepted_full_pack_manifest(
            artifact_rag_root=root,
            accepted_jsonl=accepted,
            candidates_jsonl=candidates,
            sixs_stem_manifest_json=sixs_manifest,
            output_dir=root / "packsrc" / "r001",
        )
