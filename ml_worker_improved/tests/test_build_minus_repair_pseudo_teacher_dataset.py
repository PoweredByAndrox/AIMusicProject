from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.build_minus_repair_pseudo_teacher_dataset import (
    build_minus_repair_pseudo_teacher_dataset,
    consumer_path,
    read_audio,
    sha256_file,
)
from tools.pseudo_teacher_contract import PSEUDO_TEACHER_SCHEMA


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int = 8_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_24")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _candidate(candidate_root: Path, track_id: str, *, sample_rate: int = 8_000) -> None:
    frames = sample_rate // 2
    time = np.arange(frames, dtype=np.float32) / float(sample_rate)
    base = np.stack(
        [0.16 * np.sin(2.0 * np.pi * 180.0 * time), 0.14 * np.sin(2.0 * np.pi * 187.0 * time)],
        axis=1,
    )
    vocal = np.stack(
        [0.08 * np.sin(2.0 * np.pi * 760.0 * time), 0.07 * np.sin(2.0 * np.pi * 780.0 * time)],
        axis=1,
    )
    run_dir = candidate_root / track_id / "run"
    _write_audio(run_dir / "instrumental.flac", base, sample_rate)
    _write_audio(run_dir / "vocals.flac", vocal, sample_rate)
    _write_json(
        run_dir / "stem_manifest.json",
        {
            "target_stem": "instrumental",
            "model_output_stem_type": "instrumental",
            "contract": {"fallback_used": False},
            "execution_contract": {"inference_runtime": {"vendor_cpu_fallback_detected": False}},
        },
    )


def _project_alias(mount: Path, path: Path, *, style: str) -> str:
    relative = path.relative_to(mount).as_posix()
    if style == "work":
        return f"/work/{relative}"
    if style == "consumer":
        return f"/Users/test/AIMusicProject_win/{relative}"
    raise ValueError(f"unsupported style: {style}")


def _clip_track(mount: Path, track_id: str, *, split: str = "train", sample_rate: int = 8_000) -> dict:
    frames = sample_rate // 2
    time = np.arange(frames, dtype=np.float32) / float(sample_rate)
    clip = np.stack(
        [0.09 * np.sin(2.0 * np.pi * 240.0 * time), 0.07 * np.sin(2.0 * np.pi * 244.0 * time)],
        axis=1,
    )
    clip_path = mount / "artifacts" / "clips" / track_id / "input.flac"
    _write_audio(clip_path, clip, sample_rate)
    return {
        "track_id": track_id,
        "split": split,
        "windows_clip_path": str(clip_path),
        "mac_clip_path": _project_alias(mount, clip_path, style="consumer"),
    }


def _conditioning_proxy_root(
    mount: Path,
    root_name: str,
    track_id: str,
    *,
    model_id: str,
    source_audio_path: str,
    sample_rate: int = 8_000,
    style: str = "work",
    vocal_audio: np.ndarray | None = None,
) -> Path:
    frames = sample_rate // 2
    time = np.arange(frames, dtype=np.float32) / float(sample_rate)
    vocal = vocal_audio
    if vocal is None:
        vocal = np.stack(
            [0.06 * np.sin(2.0 * np.pi * 620.0 * time), 0.05 * np.sin(2.0 * np.pi * 655.0 * time)],
            axis=1,
        )
    run_dir = mount / "artifacts" / root_name / track_id / "run"
    vocal_path = _write_audio(run_dir / "vocals.flac", vocal, sample_rate)
    _write_json(
        run_dir / "stem_manifest.json",
        {
            "source_audio_path": source_audio_path,
            "model_id": model_id,
            "target_stem": "vocals",
            "target_product_stem": "vocal",
            "target_stem_artifact": {
                "path": _project_alias(mount, vocal_path, style=style),
                "target_stem": "vocals",
                "product_stem_type": "vocal",
                "user_facing_stem_role": "vocal",
                "source": "model_output",
            },
            "stems": {
                "vocals": {
                    "path": _project_alias(mount, vocal_path, style=style),
                    "role": "target",
                    "source": "model_output",
                    "sha256": sha256_file(vocal_path),
                }
            },
        },
    )
    return mount / "artifacts" / root_name


def test_consumer_path_maps_container_mount_to_native_mount(tmp_path: Path) -> None:
    producer_root = tmp_path / "work"
    source = producer_root / "artifacts" / "mac_api" / "ds1" / "audio" / "t01" / "src.flac"

    mapped = consumer_path(
        source,
        producer_project_root=producer_root,
        consumer_project_root=Path("/Users/test/AIMusicProject_win"),
    )

    assert mapped.as_posix() == "/Users/test/AIMusicProject_win/artifacts/mac_api/ds1/audio/t01/src.flac"


def test_builds_track_disjoint_minus_repair_rows_with_short_paths(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    output_dir = mount / "artifacts" / "ds1"
    candidate_root = mount / "artifacts" / "candidates"
    mount.mkdir()
    for track_id in ("t01", "t02", "t03"):
        _candidate(candidate_root, track_id)
    clip_manifest = _write_json(
        tmp_path / "clips.json",
        {
            "tracks": [
                {"track_id": "t01", "split": "train"},
                {"track_id": "t02", "split": "train"},
                {"track_id": "t03", "split": "holdout"},
            ]
        },
    )

    manifest = build_minus_repair_pseudo_teacher_dataset(
        clip_batch_manifest=clip_manifest,
        candidate_root=candidate_root,
        output_dir=output_dir,
        mac_windows_project_mount=mount,
        windows_project_root="D:/AIMusicProject",
    )

    assert manifest["train_track_ids"] == ["t01", "t02"]
    assert manifest["holdout_track_ids"] == ["t03"]
    assert manifest["track_split_overlap"] == []
    assert manifest["train_example_count"] == 8
    assert manifest["holdout_example_count"] == 4
    assert manifest["delivery_allowed"] is False
    train_rows = _read_jsonl(output_dir / "train.jsonl")
    holdout_rows = _read_jsonl(Path(manifest["holdout_rows_jsonl"]))
    assert all(row["schema"] == PSEUDO_TEACHER_SCHEMA for row in train_rows + holdout_rows)
    assert all(row["trainable"] is True for row in train_rows)
    assert all(row["trainable"] is False for row in holdout_rows)
    assert all(len(Path(row["source_audio_path"]).name) <= 8 for row in train_rows + holdout_rows)
    leak_row = next(row for row in train_rows if row["corruption"]["variant"] == "vocal_leak")
    source, _ = read_audio(Path(leak_row["source_audio_path"]))
    target, _ = read_audio(Path(leak_row["target_audio_path"]))
    assert float(np.mean(np.square(source - target))) > 1.0e-7


def test_rejects_candidate_sample_rate_mismatch(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    output_dir = mount / "artifacts" / "ds1"
    candidate_root = tmp_path / "candidates"
    mount.mkdir()
    _candidate(candidate_root, "t01")
    vocal_path = candidate_root / "t01" / "run" / "vocals.flac"
    _write_audio(vocal_path, np.zeros((4000, 2), dtype=np.float32), sample_rate=16_000)
    clip_manifest = _write_json(tmp_path / "clips.json", {"tracks": [{"track_id": "t01", "split": "train"}]})

    with pytest.raises(ValueError, match="sample-rate mismatch"):
        build_minus_repair_pseudo_teacher_dataset(
            clip_batch_manifest=clip_manifest,
            candidate_root=candidate_root,
            output_dir=output_dir,
            mac_windows_project_mount=mount,
        )


def test_attaches_sorted_conditioning_proxies_from_multiple_roots(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    output_dir = mount / "artifacts" / "ds1"
    candidate_root = mount / "artifacts" / "candidates"
    _candidate(candidate_root, "t01")
    clip_track = _clip_track(mount, "t01")
    clip_manifest = _write_json(tmp_path / "clips.json", {"tracks": [clip_track]})
    p1_root = _conditioning_proxy_root(
        mount,
        "p1",
        "t01",
        model_id="proxy_a",
        source_audio_path=_project_alias(mount, mount / "artifacts" / "clips" / "t01" / "input.flac", style="consumer"),
        style="consumer",
    )
    p2_root = _conditioning_proxy_root(
        mount,
        "p2",
        "t01",
        model_id="proxy_b",
        source_audio_path=_project_alias(mount, mount / "artifacts" / "clips" / "t01" / "input.flac", style="work"),
        style="work",
        vocal_audio=np.stack(
            [
                0.05 * np.sin(2.0 * np.pi * 710.0 * np.arange(4_000, dtype=np.float32) / 8_000.0),
                0.04 * np.sin(2.0 * np.pi * 745.0 * np.arange(4_000, dtype=np.float32) / 8_000.0),
            ],
            axis=1,
        ),
    )
    p2_manifest_path = p2_root / "t01" / "run" / "stem_manifest.json"
    p2_manifest = json.loads(p2_manifest_path.read_text(encoding="utf-8"))
    p2_manifest.pop("target_stem")
    p2_manifest.pop("target_product_stem")
    p2_manifest["target_stem_artifact"].pop("target_stem")
    p2_manifest["target_stem_artifact"].pop("product_stem_type")
    p2_manifest["target_stem_artifact"].pop("user_facing_stem_role")
    p2_manifest["target_stem_artifact"].pop("source")
    p2_manifest["stems"]["vocals"].pop("source")
    p2_manifest["model"] = {"model_id": "proxy_b", "target_output": "vocals"}
    _write_json(p2_manifest_path, p2_manifest)

    manifest = build_minus_repair_pseudo_teacher_dataset(
        clip_batch_manifest=clip_manifest,
        candidate_root=candidate_root,
        output_dir=output_dir,
        mac_windows_project_mount=mount,
        consumer_project_root=Path("/Users/test/AIMusicProject_win"),
        conditioning_proxy_roots=[p2_root, p1_root],
        windows_project_root="D:/AIMusicProject",
    )

    train_rows = _read_jsonl(output_dir / "train.jsonl")
    assert train_rows
    expected_paths = [
        "/Users/test/AIMusicProject_win/artifacts/p1/t01/run/vocals.flac",
        "/Users/test/AIMusicProject_win/artifacts/p2/t01/run/vocals.flac",
    ]
    for row in train_rows:
        assert row["conditioning_model_ids"] == ["proxy_a", "proxy_b"]
        assert [Path(path).as_posix() for path in row["conditioning_audio_paths"]] == expected_paths
    track_record = manifest["tracks"][0]
    assert track_record["conditioning_proxy_count"] == 2
    assert track_record["conditioning_model_ids"] == ["proxy_a", "proxy_b"]
    assert len(track_record["conditioning_vocal_sha256"]) == 2
    assert track_record["conditioning_source_provenance"] == ["model_output", "model_target_output:proxy_b"]
    assert manifest["conditioning_proxy_root_count"] == 2
    assert manifest["conditioning_track_count"] == 1
    assert manifest["delivery_allowed"] is False


@pytest.mark.parametrize(
    ("proxy_a_model", "proxy_b_model", "reuse_audio", "error_match"),
    [
        ("dup_model", "dup_model", False, "duplicate conditioning proxy model id"),
        ("proxy_a", "proxy_b", True, "duplicate conditioning proxy vocal sha256"),
    ],
)
def test_rejects_duplicate_conditioning_proxy_model_or_audio(
    tmp_path: Path,
    proxy_a_model: str,
    proxy_b_model: str,
    reuse_audio: bool,
    error_match: str,
) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    output_dir = mount / "artifacts" / "ds1"
    candidate_root = mount / "artifacts" / "candidates"
    _candidate(candidate_root, "t01")
    clip_track = _clip_track(mount, "t01")
    clip_manifest = _write_json(tmp_path / "clips.json", {"tracks": [clip_track]})
    base_audio = np.ones((4_000, 2), dtype=np.float32) * 0.05
    p1_root = _conditioning_proxy_root(
        mount,
        "p1",
        "t01",
        model_id=proxy_a_model,
        source_audio_path=_project_alias(mount, mount / "artifacts" / "clips" / "t01" / "input.flac", style="work"),
        vocal_audio=base_audio,
    )
    second_audio = base_audio if reuse_audio else base_audio * 0.5
    p2_root = _conditioning_proxy_root(
        mount,
        "p2",
        "t01",
        model_id=proxy_b_model,
        source_audio_path=_project_alias(mount, mount / "artifacts" / "clips" / "t01" / "input.flac", style="consumer"),
        style="consumer",
        vocal_audio=second_audio,
    )

    with pytest.raises(ValueError, match=error_match):
        build_minus_repair_pseudo_teacher_dataset(
            clip_batch_manifest=clip_manifest,
            candidate_root=candidate_root,
            output_dir=output_dir,
            mac_windows_project_mount=mount,
            consumer_project_root=Path("/Users/test/AIMusicProject_win"),
            conditioning_proxy_roots=[p1_root, p2_root],
            windows_project_root="D:/AIMusicProject",
        )


def test_rejects_conditioning_proxy_from_wrong_source_track(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    output_dir = mount / "artifacts" / "ds1"
    candidate_root = mount / "artifacts" / "candidates"
    _candidate(candidate_root, "t01")
    clip_track = _clip_track(mount, "t01")
    clip_manifest = _write_json(tmp_path / "clips.json", {"tracks": [clip_track]})
    wrong_source = "/work/artifacts/clips/t99/input.flac"
    proxy_root = _conditioning_proxy_root(
        mount,
        "p1",
        "t01",
        model_id="proxy_a",
        source_audio_path=wrong_source,
    )

    with pytest.raises(ValueError, match="conditioning proxy source mismatch"):
        build_minus_repair_pseudo_teacher_dataset(
            clip_batch_manifest=clip_manifest,
            candidate_root=candidate_root,
            output_dir=output_dir,
            mac_windows_project_mount=mount,
            conditioning_proxy_roots=[proxy_root],
            windows_project_root="D:/AIMusicProject",
        )
