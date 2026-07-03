from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import tools.run_fully_automatic_stem_search as stem_search
from tools.run_fully_automatic_stem_search import (
    _candidate_contamination_checks,
    _copy_target_specific_outputs,
    _discover_default_contamination_model_manifest,
    _parse_teacher_mask_checkpoint_spec,
    apply_minus_vocal_debleed,
    apply_stem_merges,
    build_candidate_manifest,
    build_default_from_scratch_recipe,
    build_model_split_v2_recipe,
    choose_next_action,
    run_fully_automatic_stem_search,
    select_accepted_candidate,
)
from tools.stem_contamination_common import (
    FORBIDDEN_SOURCES,
    STEMS,
    VERDICTS,
    missing_required_stems,
    validate_no_artifact_rag_output,
)
from tools.stem_contamination_model import band_limited_audio
from tools.train_stem_contamination_detector import train_stem_contamination_detector
from tools.stem_region_repair_common import ISSUE_BAD_RECONSTRUCTION, ISSUE_STEREO_COLLAPSE, make_region_mask
from tools.stem_region_repair_model import (
    StemRegionRepairTrainConfig,
    save_region_repair_checkpoint,
    train_region_repair_model_from_arrays,
)


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float = 0.25, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def bright_voice_like_vocal(seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / float(SAMPLE_RATE)
    f0 = 155.0 + 24.0 * np.sin(2.0 * math.pi * 5.3 * t)
    phase = np.cumsum(2.0 * math.pi * f0 / float(SAMPLE_RATE)).astype(np.float32)
    audio = np.zeros_like(t)
    for harmonic in range(1, 18):
        audio += (0.12 / float(harmonic) ** 0.82) * np.sin(float(harmonic) * phase + 0.11 * harmonic).astype(np.float32)
    return (audio * (0.46 + 0.54 * np.sin(2.0 * math.pi * 2.7 * t) ** 2)).astype(np.float32)


def cymbal_like_drums(seconds: float = 0.5) -> np.ndarray:
    frames = int(SAMPLE_RATE * seconds)
    audio = np.zeros(frames, dtype=np.float32)
    rng = np.random.default_rng(4242)
    pulse_spacing = int(SAMPLE_RATE * 0.16)
    decay_len = int(SAMPLE_RATE * 0.05)
    decay = np.exp(-np.arange(decay_len, dtype=np.float32) / float(SAMPLE_RATE * 0.008))
    for start in range(0, frames, pulse_spacing):
        end = min(frames, start + decay_len)
        audio[start:end] += rng.normal(0.0, 0.25, end - start).astype(np.float32) * decay[: end - start]
    return audio.astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return path


def write_stereo_wav(path: Path, left: np.ndarray, right: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.stack([left, right], axis=1).astype(np.float32)
    sf.write(path, audio, SAMPLE_RATE)
    return path


def write_full_foreign_references(tmp_path: Path, source_audio: dict[str, np.ndarray]) -> dict[str, dict[str, str]]:
    ref_paths = {
        stem: write_wav(tmp_path / "refs" / f"{stem}.flac", audio)
        for stem, audio in source_audio.items()
    }
    return {
        tested_stem: {foreign_source: str(ref_paths[foreign_source]) for foreign_source in foreign_sources}
        for tested_stem, foreign_sources in FORBIDDEN_SOURCES.items()
    }


def find_leak_result(candidate: dict, tested_stem: str, foreign_source: str) -> dict:
    return next(
        result
        for result in candidate["contamination_score"]["leak_results"]
        if result["tested_stem"] == tested_stem and result["foreign_source"] == foreign_source
    )


def test_apply_stem_merges_moves_other_into_guitar(tmp_path: Path) -> None:
    guitar_audio = sine(330.0, amp=0.05)
    other_audio = sine(550.0, amp=0.02)
    guitar_path = write_wav(tmp_path / "guitar.wav", guitar_audio)
    other_path = write_wav(tmp_path / "other.wav", other_audio)

    result = apply_stem_merges(
        stem_paths={"guitar": str(guitar_path), "other": str(other_path)},
        merge_specs=[{"source_stem": "other", "target_stem": "guitar", "mode": "move"}],
    )

    merged_guitar, _ = sf.read(guitar_path, dtype="float32", always_2d=False)
    emptied_other, _ = sf.read(other_path, dtype="float32", always_2d=False)

    assert result["applied"] is True
    assert result["blocker_reasons"] == []
    assert result["operations"] == [
        {
            "source_stem": "other",
            "target_stem": "guitar",
            "mode": "move",
            "source_path": str(other_path),
            "target_path": str(guitar_path),
        }
    ]
    np.testing.assert_allclose(merged_guitar, guitar_audio + other_audio, atol=1e-4)
    np.testing.assert_allclose(emptied_other, np.zeros_like(other_audio), atol=1e-4)


def test_all_six_stems_are_required() -> None:
    assert STEMS == ("vocals", "drums", "bass", "guitar", "piano", "other")
    assert missing_required_stems({"vocals": "v.flac", "drums": "d.flac"}) == [
        "bass",
        "guitar",
        "piano",
        "other",
    ]


def test_no_human_review_state_exists() -> None:
    assert "human_review" not in VERDICTS
    assert all("human" not in verdict for verdict in VERDICTS)
    assert "vocals" in FORBIDDEN_SOURCES["minus"]


def test_unknown_blocks_instead_of_review() -> None:
    action = choose_next_action({"verdict": "blocked_unknown", "reason": "detector_confidence_insufficient"})

    assert action["action"] == "block"
    assert action["verdict"] == "blocked_unknown"
    assert "human" not in json.dumps(action)


def test_candidate_untrusted_by_default(tmp_path: Path) -> None:
    manifest = build_candidate_manifest(candidate_id="c001", output_dir=tmp_path / "c001")

    assert manifest["candidate_id"] == "c001"
    assert manifest["candidate_untrusted"] is True
    assert manifest["verdict"] == "blocked_unknown"
    assert manifest["minus_rebuilt_from_non_vocal_stems"] is False


def test_cli_enables_live_progress_echo_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    captured: dict[str, str] = {}

    def fake_run_fully_automatic_stem_search(**_kwargs):
        captured["echo_progress"] = stem_search.os.environ.get("STEM_SEARCH_ECHO_PROGRESS", "")
        return {"verdict": "blocked_unknown"}

    monkeypatch.delenv("STEM_SEARCH_ECHO_PROGRESS", raising=False)
    monkeypatch.setattr(stem_search, "run_fully_automatic_stem_search", fake_run_fully_automatic_stem_search)
    monkeypatch.setattr(
        stem_search.sys,
        "argv",
        [
            "run_fully_automatic_stem_search.py",
            "--mix",
            str(tmp_path / "mix.flac"),
            "--out",
            str(tmp_path / "out"),
            "--no-artifact-rag",
        ],
    )

    stem_search.main()
    capsys.readouterr()

    assert captured["echo_progress"] == "1"


def test_cli_can_disable_live_progress_echo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    captured: dict[str, str] = {}

    def fake_run_fully_automatic_stem_search(**_kwargs):
        captured["echo_progress"] = stem_search.os.environ.get("STEM_SEARCH_ECHO_PROGRESS", "")
        return {"verdict": "blocked_unknown"}

    monkeypatch.setenv("STEM_SEARCH_ECHO_PROGRESS", "1")
    monkeypatch.setattr(stem_search, "run_fully_automatic_stem_search", fake_run_fully_automatic_stem_search)
    monkeypatch.setattr(
        stem_search.sys,
        "argv",
        [
            "run_fully_automatic_stem_search.py",
            "--mix",
            str(tmp_path / "mix.flac"),
            "--out",
            str(tmp_path / "out"),
            "--no-artifact-rag",
            "--no-echo-progress",
        ],
    )

    stem_search.main()
    capsys.readouterr()

    assert captured["echo_progress"] == "0"


def test_cli_passes_gpu_wait_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    captured: dict[str, float] = {}

    def fake_run_fully_automatic_stem_search(**kwargs):
        captured["wait_for_gpu_free_sec"] = kwargs["wait_for_gpu_free_sec"]
        captured["gpu_preflight_poll_sec"] = kwargs["gpu_preflight_poll_sec"]
        return {"verdict": "blocked_unknown"}

    monkeypatch.setattr(stem_search, "run_fully_automatic_stem_search", fake_run_fully_automatic_stem_search)
    monkeypatch.setattr(
        stem_search.sys,
        "argv",
        [
            "run_fully_automatic_stem_search.py",
            "--mix",
            str(tmp_path / "mix.flac"),
            "--out",
            str(tmp_path / "out"),
            "--no-artifact-rag",
            "--wait-for-gpu-free-sec",
            "30",
            "--gpu-preflight-poll-sec",
            "2.5",
        ],
    )

    stem_search.main()
    capsys.readouterr()

    assert captured == {"wait_for_gpu_free_sec": 30.0, "gpu_preflight_poll_sec": 2.5}


def test_gpu_preflight_blocks_busy_gpu_before_candidate_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_collect_cuda_preflight(*, require_gpu: bool, min_free_mib: int) -> dict:
        return {
            "schema": "stem_search_cuda_preflight.v1",
            "require_gpu": require_gpu,
            "min_free_mib": min_free_mib,
            "status": "blocked",
            "reason": "gpu_free_memory_too_low",
            "free_mib": 512,
            "total_mib": 12_288,
            "used_mib": 11_776,
        }

    monkeypatch.setattr(stem_search, "_collect_cuda_preflight", fake_collect_cuda_preflight)

    result = run_fully_automatic_stem_search(
        mix_path=tmp_path / "mix.flac",
        output_dir=tmp_path / "run",
        require_gpu=True,
        no_artifact_rag=True,
        allow_default_recipe=False,
        min_gpu_free_mib=4096,
    )
    progress_rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert result["verdict"] == "blocked_unknown"
    assert result["root_cause"] == "infra_gap"
    assert result["failed_checks"] == ["gpu_preflight_failed"]
    assert result["gpu_preflight"]["reason"] == "gpu_free_memory_too_low"
    assert result["candidate_count"] == 0
    assert [row["event"] for row in progress_rows] == ["start", "gpu_preflight", "end"]
    assert not any(row["event"] == "candidate_start" for row in progress_rows)


def test_gpu_preflight_block_manifest_keeps_root_cause_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = "/repo/artifacts/stem_eval/t337_recon_synth/model_vocals/model.pt"
    queue_jsonl = tmp_path / "queue.jsonl"
    queue_jsonl.write_text(
        json.dumps(
            {
                "row_id": "rc000001",
                "root_cause": "repair_gap",
                "issue_type": "bad_reconstruction",
                "failed_repair_id": "probe_repair",
                "failed_repair_checkpoint": checkpoint,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_collect_cuda_preflight(*, require_gpu: bool, min_free_mib: int) -> dict:
        return {
            "schema": "stem_search_cuda_preflight.v1",
            "require_gpu": require_gpu,
            "min_free_mib": min_free_mib,
            "status": "blocked",
            "reason": "gpu_free_memory_too_low",
            "free_mib": 512,
            "total_mib": 12_288,
            "used_mib": 11_776,
        }

    monkeypatch.setattr(stem_search, "_collect_cuda_preflight", fake_collect_cuda_preflight)

    result = run_fully_automatic_stem_search(
        mix_path=tmp_path / "mix.flac",
        output_dir=tmp_path / "run",
        require_gpu=True,
        no_artifact_rag=True,
        allow_default_recipe=False,
        root_cause_queue_jsonl=queue_jsonl,
        min_gpu_free_mib=4096,
    )

    assert result["verdict"] == "blocked_unknown"
    assert result["root_cause_route_policy"]["source_queue_jsonl"] == str(queue_jsonl)
    assert result["root_cause_route_policy"]["blocked_repair_ids"] == ["probe_repair"]
    assert result["root_cause_route_policy"]["blocked_repair_checkpoints"] == [checkpoint]


def test_gpu_preflight_waits_for_free_memory_before_candidate_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    preflight_rows = [
        {
            "schema": "stem_search_cuda_preflight.v1",
            "require_gpu": True,
            "min_free_mib": 4096,
            "status": "blocked",
            "reason": "gpu_free_memory_too_low",
            "free_mib": 1024,
            "total_mib": 12_288,
            "used_mib": 11_264,
        },
        {
            "schema": "stem_search_cuda_preflight.v1",
            "require_gpu": True,
            "min_free_mib": 4096,
            "status": "passed",
            "reason": "",
            "free_mib": 9000,
            "total_mib": 12_288,
            "used_mib": 3288,
        },
    ]
    calls: list[int] = []

    def fake_collect_cuda_preflight(*, require_gpu: bool, min_free_mib: int) -> dict:
        calls.append(min_free_mib)
        return dict(preflight_rows[min(len(calls) - 1, len(preflight_rows) - 1)])

    monkeypatch.setattr(stem_search, "_collect_cuda_preflight", fake_collect_cuda_preflight)
    monkeypatch.setattr(stem_search.time, "sleep", lambda _seconds: None)

    result = run_fully_automatic_stem_search(
        mix_path=tmp_path / "mix.flac",
        output_dir=tmp_path / "run",
        require_gpu=True,
        no_artifact_rag=True,
        allow_default_recipe=False,
        min_gpu_free_mib=4096,
        wait_for_gpu_free_sec=30,
        gpu_preflight_poll_sec=5,
    )
    progress_rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(calls) == 2
    assert result["gpu_preflight"]["status"] == "passed"
    assert any(row["event"] == "gpu_preflight_wait" for row in progress_rows)
    assert any(row["event"] == "gpu_preflight" and row["status"] == "passed" for row in progress_rows)
    assert not any(row["event"] == "candidate_start" for row in progress_rows)


def test_root_cause_route_policy_skips_quarantined_and_prioritizes_repairable(tmp_path: Path) -> None:
    queue_jsonl = tmp_path / "queue.jsonl"
    queue_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "row_id": "rc000001",
                        "issue_type": "infra_failure",
                        "route_signature": "vocal_lane:broken_vocal_model",
                        "quarantined_for_next_run": True,
                    }
                ),
                json.dumps(
                    {
                        "row_id": "rc000002",
                        "issue_type": "infra_failure",
                        "route_signature": "minus_lane:bs_mamba2_vocals",
                        "quarantined_for_next_run": True,
                    }
                ),
                json.dumps(
                    {
                        "row_id": "rc000003",
                        "issue_type": "bad_reconstruction",
                        "affected_stem": "drums",
                        "route_signature": "stem_lane:repairable_stem_model",
                        "quarantined_for_next_run": False,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    recipe = {
        "schema": "stem_search_recipe.v2",
        "candidates": [
            {
                "candidate_id": "c001",
                "target_specific_commands": [{"lane": "vocal_lane", "route_id": "broken_vocal_model"}],
            },
            {
                "candidate_id": "c002",
                "target_specific_commands": [{"lane": "stem_lane", "route_id": "unseen_model"}],
            },
            {
                "candidate_id": "c003",
                "target_specific_commands": [{"lane": "stem_lane", "route_id": "repairable_stem_model"}],
            },
            {
                "candidate_id": "c004",
                "target_specific_commands": [{"lane": "minus_lane", "route_id": "bs_mamba2_vocals:instrumental"}],
            },
        ],
    }

    filtered = stem_search._apply_root_cause_route_policy(recipe, root_cause_queue_jsonl=queue_jsonl)

    assert [candidate["candidate_id"] for candidate in filtered["candidates"]] == ["c003", "c002"]
    assert filtered["root_cause_route_policy"]["skipped_candidate_count"] == 2
    assert filtered["root_cause_route_policy"]["prioritized_candidate_count"] == 1
    assert filtered["root_cause_route_policy"]["skipped_candidates"][0]["candidate_id"] == "c001"
    assert filtered["root_cause_route_policy"]["skipped_candidates"][1]["candidate_id"] == "c004"


def test_root_cause_route_policy_filters_failed_repair_checkpoint(tmp_path: Path) -> None:
    failed_checkpoint = "/repo/artifacts/stem_eval/t337_recon_synth/model_vocals/model.pt"
    safe_checkpoint = "/repo/artifacts/stem_eval/t334_bass_synth/model/model.pt"
    queue_jsonl = tmp_path / "queue.jsonl"
    queue_jsonl.write_text(
        json.dumps(
            {
                "row_id": "rc000001",
                "root_cause": "repair_gap",
                "issue_type": "bad_reconstruction",
                "affected_stem": "vocals",
                "training_action": "retrain_failed_repair_checkpoint",
                "route_signature": "stem_lane:htdemucs_6s|vocal_lane:mdx23c_vocals",
                "failed_repair_id": "srseq_bad",
                "failed_repair_checkpoint": failed_checkpoint,
                "quarantined_for_next_run": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recipe = {
        "schema": "stem_search_recipe.v2",
        "candidates": [
            {
                "candidate_id": "c001",
                "target_specific_commands": [
                    {"lane": "stem_lane", "route_id": "htdemucs_6s"},
                    {"lane": "vocal_lane", "route_id": "mdx23c_vocals"},
                ],
                "model_split_composite_repair_options": [
                    {
                        "repair_id": "srseq_bad",
                        "type": "stem_region_repair",
                        "stem_region_repairs": [
                            {
                                "stem": "vocals",
                                "issue_type": "bad_reconstruction",
                                "checkpoint": failed_checkpoint,
                            }
                        ],
                    },
                    {
                        "repair_id": "srseq_safe",
                        "type": "stem_region_repair",
                        "stem_region_repairs": [
                            {
                                "stem": "bass",
                                "issue_type": "stereo_collapse",
                                "checkpoint": safe_checkpoint,
                            }
                        ],
                    },
                ],
            }
        ],
    }

    filtered = stem_search._apply_root_cause_route_policy(recipe, root_cause_queue_jsonl=queue_jsonl)
    repair_options = filtered["candidates"][0]["model_split_composite_repair_options"]

    assert [option["repair_id"] for option in repair_options] == ["srseq_safe"]
    assert filtered["root_cause_route_policy"]["blocked_repair_ids"] == ["srseq_bad"]
    assert filtered["root_cause_route_policy"]["blocked_repair_checkpoints"] == [failed_checkpoint]
    assert filtered["root_cause_route_policy"]["skipped_repair_option_count"] == 1


def test_root_cause_route_policy_inspection_writes_diagnostic_manifest(tmp_path: Path) -> None:
    from tools.inspect_root_cause_route_policy import _manifest_cli_summary, inspect_root_cause_route_policy

    failed_checkpoint = "/repo/artifacts/stem_eval/t337_recon_synth/model_vocals/model.pt"
    safe_checkpoint = "/repo/artifacts/stem_eval/t334_bass_synth/model/model.pt"
    queue_jsonl = tmp_path / "queue.jsonl"
    queue_jsonl.write_text(
        json.dumps(
            {
                "row_id": "rc000001",
                "root_cause": "repair_gap",
                "issue_type": "bad_reconstruction",
                "affected_stem": "vocals",
                "training_action": "retrain_failed_repair_checkpoint",
                "failed_repair_id": "srseq_bad",
                "failed_repair_checkpoint": failed_checkpoint,
                "quarantined_for_next_run": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "target_specific_commands": [{"lane": "vocal_lane", "route_id": "mdx23c_vocals"}],
                        "model_split_composite_repair_options": [
                            {
                                "repair_id": "srseq_bad",
                                "type": "stem_region_repair",
                                "stem_region_repairs": [
                                    {
                                        "stem": "vocals",
                                        "issue_type": "bad_reconstruction",
                                        "checkpoint": failed_checkpoint,
                                    }
                                ],
                            },
                            {
                                "repair_id": "srseq_safe",
                                "type": "stem_region_repair",
                                "stem_region_repairs": [
                                    {
                                        "stem": "bass",
                                        "issue_type": "stereo_collapse",
                                        "checkpoint": safe_checkpoint,
                                    }
                                ],
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = inspect_root_cause_route_policy(
        output_dir=tmp_path / "policy",
        root_cause_queue_jsonl=queue_jsonl,
        recipe_path=recipe_path,
        no_artifact_rag=True,
    )

    assert manifest["delivery_allowed"] is False
    assert manifest["delivery_status"] == "blocked"
    assert manifest["presentation_status"] == "diagnostic_policy_only"
    assert manifest["listen_pack"] == {}
    assert manifest["candidate_count_before_policy"] == 1
    assert manifest["candidate_count_after_policy"] == 1
    assert manifest["root_cause_route_policy"]["blocked_repair_ids"] == ["srseq_bad"]
    assert manifest["root_cause_route_policy"]["blocked_repair_checkpoints"] == [failed_checkpoint]
    assert manifest["root_cause_route_policy"]["skipped_repair_option_count"] == 1
    assert manifest["remaining_repair_option_count"] == 1
    assert manifest["active_blocked_repair_option_count"] == 0
    assert manifest["active_blocked_repair_options"] == []
    assert (tmp_path / "policy" / "manifest.json").exists()
    progress_events = [
        json.loads(line)["event"]
        for line in (tmp_path / "policy" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert progress_events == ["start", "end"]
    summary = _manifest_cli_summary(manifest)
    assert "skipped_repair_options" not in summary
    assert summary["skipped_repair_option_count"] == 1
    assert summary["active_blocked_repair_option_count"] == 0
    assert summary["manifest_path"].endswith("manifest.json")


def test_build_stem_search_launch_manifest_records_preflight_and_command(tmp_path: Path) -> None:
    from tools.build_stem_search_launch_manifest import build_stem_search_launch_manifest

    recipe_path = tmp_path / "recipe_after_policy.json"
    recipe_path.write_text(json.dumps({"schema": "stem_search_recipe.v2", "candidates": []}), encoding="utf-8")
    mix_path = Path("/repo/testing_stuff_music/source.mp3")
    output_dir = Path("/repo/artifacts/stem_eval/t999_run")
    preflight = {
        "schema": "stem_search_cuda_preflight.v1",
        "require_gpu": True,
        "min_free_mib": 4096,
        "status": "blocked",
        "reason": "gpu_free_memory_too_low",
        "free_mib": 1024,
        "total_mib": 12282,
        "used_mib": 11258,
    }
    resource_snapshot = {
        "schema": "stem_search_resource_blocker_snapshot.v1",
        "docker_container_stats_available": True,
        "docker_container_stats_reason": "",
        "top_container_memory": [
            {
                "name": "trading-vllm-server",
                "cpu_percent": "105.99%",
                "mem_usage": "2.901GiB / 23.46GiB",
                "mem_percent": "12.36%",
            }
        ],
        "gpu_processes": [{"pid": "41720", "process_name": "Codex.exe", "used_memory_mib": ""}],
    }

    manifest = build_stem_search_launch_manifest(
        output_dir=tmp_path / "launch",
        mix_path=mix_path,
        recipe_path=recipe_path,
        run_output_dir=output_dir,
        max_candidates=10,
        require_gpu=True,
        min_gpu_free_mib=4096,
        wait_for_gpu_free_sec=600.0,
        gpu_preflight=preflight,
        resource_snapshot=resource_snapshot,
        no_artifact_rag=True,
    )

    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "diagnostic_launch_only"
    assert manifest["listen_pack"] == {}
    assert manifest["gpu_ready"] is False
    assert manifest["gpu_preflight"]["reason"] == "gpu_free_memory_too_low"
    assert manifest["gpu_preflight_summary"]["status"] == "blocked"
    assert manifest["gpu_preflight_summary"]["free_mib"] == 1024
    assert manifest["status_summary"]["result_status"] == "not_started"
    assert manifest["status_summary"]["blocked_reason"] == "gpu_free_memory_too_low"
    assert manifest["resource_blocker_snapshot"]["top_container_memory"][0]["name"] == "trading-vllm-server"
    assert manifest["resource_blocker_summary"]["top_container_memory_names"] == ["trading-vllm-server"]
    assert manifest["recipe_sha256"]
    assert manifest["launch_status"] == "blocked_gpu_preflight"
    assert manifest["in_container_command"][:2] == ["python", "tools/run_fully_automatic_stem_search.py"]
    assert "--recipe" in manifest["in_container_command"]
    assert str(recipe_path) in manifest["in_container_command"]
    assert "--wait-for-gpu-free-sec" in manifest["in_container_command"]
    assert manifest["docker_compose_command_host"][:4] == ["docker", "compose", "run", "--rm"]
    assert "-v" in manifest["docker_compose_command_host"]
    socket_mount_index = manifest["docker_compose_command_host"].index("-v") + 1
    assert manifest["docker_compose_command_host"][socket_mount_index] == "/var/run/docker.sock:/var/run/docker.sock"
    progress_events = [
        json.loads(line)["event"]
        for line in (tmp_path / "launch" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert progress_events == ["start", "resource_snapshot_start", "resource_snapshot_end", "end"]
    summary_md = (tmp_path / "launch" / "summary.md").read_text(encoding="utf-8")
    assert "GPU preflight: blocked" in summary_md
    assert "free_mib=1024" in summary_md
    assert "Top container memory: trading-vllm-server" in summary_md
    assert "Result: not started" in summary_md


def test_resource_snapshot_ignores_nvidia_not_found_rows() -> None:
    from tools.build_stem_search_launch_manifest import _parse_gpu_process_rows

    rows = _parse_gpu_process_rows(
        "[Not Found], [Not Found], [Not Found]\n"
        "204, [Not Found], [N/A]\n"
        "4, [Insufficient Permissions], [N/A]\n"
    )

    assert rows == [{"pid": "4", "process_name": "[Insufficient Permissions]", "used_memory_mib": "[N/A]"}]


def test_docker_socket_stats_fallback_reports_top_container_memory() -> None:
    from tools.build_stem_search_launch_manifest import _collect_docker_socket_stats

    def fake_api_get(path: str):
        if path == "/containers/json?all=0":
            return [
                {
                    "Id": "abc123",
                    "Names": ["/trading-vllm-server"],
                    "Image": "vllm/vllm-openai:latest",
                    "Status": "Up 4 hours",
                },
                {
                    "Id": "def456",
                    "Names": ["/music-postgres"],
                    "Image": "postgres:16-alpine",
                    "Status": "Up 44 hours",
                },
            ]
        if path == "/containers/abc123/stats?stream=false":
            return {"memory_stats": {"usage": 3 * 1024 * 1024 * 1024, "limit": 24 * 1024 * 1024 * 1024}}
        if path == "/containers/def456/stats?stream=false":
            return {"memory_stats": {"usage": 32 * 1024 * 1024, "limit": 24 * 1024 * 1024 * 1024}}
        raise AssertionError(path)

    result = _collect_docker_socket_stats(api_get=fake_api_get)

    assert result["available"] is True
    assert result["reason"] == ""
    assert result["top_container_memory"][0]["name"] == "trading-vllm-server"
    assert result["top_container_memory"][0]["mem_usage"] == "3.000GiB / 24.000GiB"
    assert result["top_container_memory"][0]["mem_percent"] == "12.50%"
    assert result["container_states"][0]["name"] == "trading-vllm-server"


def test_resource_snapshot_uses_docker_socket_when_docker_cli_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.build_stem_search_launch_manifest as launch_manifest

    def fake_run_text_command(command: list[str], *, timeout_sec: float = 10.0) -> dict:
        if command[0] == "nvidia-smi":
            return {
                "available": True,
                "reason": "",
                "stdout": "41720, Codex.exe, [N/A]\n",
                "stderr": "",
                "returncode": 0,
            }
        if command[0] == "docker":
            return {
                "available": False,
                "reason": "docker_cli_unavailable",
                "stdout": "",
                "stderr": "",
                "returncode": None,
            }
        raise AssertionError(command)

    monkeypatch.setattr(launch_manifest, "_run_text_command", fake_run_text_command)
    monkeypatch.setattr(
        launch_manifest,
        "_collect_docker_socket_stats",
        lambda: {
            "available": True,
            "reason": "",
            "source": "docker_engine_socket",
            "top_container_memory": [
                {
                    "name": "trading-vllm-server",
                    "cpu_percent": "",
                    "mem_usage": "3.000GiB / 24.000GiB",
                    "mem_percent": "12.50%",
                }
            ],
            "container_states": [{"name": "trading-vllm-server", "image": "vllm/vllm-openai:latest", "status": "Up"}],
        },
    )

    snapshot = launch_manifest._collect_resource_blocker_snapshot()
    summary = launch_manifest._resource_blocker_summary(snapshot)

    assert snapshot["docker_container_stats_available"] is True
    assert snapshot["docker_container_stats_reason"] == ""
    assert snapshot["docker_container_stats_source"] == "docker_engine_socket"
    assert summary["top_container_memory_names"] == ["trading-vllm-server"]


def test_run_stem_search_launch_manifest_blocks_without_running_when_gpu_busy(tmp_path: Path) -> None:
    from tools.build_stem_search_launch_manifest import build_stem_search_launch_manifest
    from tools.run_stem_search_launch_manifest import run_stem_search_launch_manifest

    recipe_path = tmp_path / "recipe_after_policy.json"
    recipe_path.write_text(json.dumps({"schema": "stem_search_recipe.v2", "candidates": []}), encoding="utf-8")
    launch_manifest = build_stem_search_launch_manifest(
        output_dir=tmp_path / "launch",
        mix_path=Path("/repo/testing_stuff_music/source.mp3"),
        recipe_path=recipe_path,
        run_output_dir=Path("/repo/artifacts/stem_eval/t999_run"),
        wait_for_gpu_free_sec=0.0,
        gpu_preflight={
            "schema": "stem_search_cuda_preflight.v1",
            "require_gpu": True,
            "min_free_mib": 4096,
            "status": "blocked",
            "reason": "gpu_free_memory_too_low",
            "free_mib": 1024,
            "total_mib": 12282,
            "used_mib": 11258,
        },
    )
    calls: list[list[str]] = []

    result = run_stem_search_launch_manifest(
        launch_manifest_path=tmp_path / "launch" / "manifest.json",
        output_dir=tmp_path / "exec",
        command_runner=lambda command, timeout_sec: calls.append(command),
        gpu_preflight_collector=lambda require_gpu, min_free_mib: {
            "schema": "stem_search_cuda_preflight.v1",
            "require_gpu": require_gpu,
            "min_free_mib": min_free_mib,
            "status": "blocked",
            "reason": "gpu_free_memory_too_low",
            "free_mib": 2048,
            "total_mib": 12282,
            "used_mib": 10234,
        },
        resource_snapshot_collector=lambda: {
            "schema": "stem_search_resource_blocker_snapshot.v1",
            "docker_container_stats_available": False,
            "docker_container_stats_reason": "docker_cli_unavailable",
            "top_container_memory": [],
            "gpu_processes": [{"pid": "4", "process_name": "Insufficient Permissions", "used_memory_mib": ""}],
        },
        no_artifact_rag=True,
    )

    assert launch_manifest["delivery_allowed"] is False
    assert calls == []
    assert result["execution_status"] == "blocked_gpu_preflight"
    assert result["delivery_allowed"] is False
    assert result["presentation_status"] == "diagnostic_launch_execution_only"
    assert result["listen_pack"] == {}
    assert result["recipe_sha256_verified"] is True
    assert result["run_command_started"] is False
    assert result["gpu_preflight_summary"]["status"] == "blocked"
    assert result["gpu_preflight_summary"]["free_mib"] == 2048
    assert result["status_summary"]["result_status"] == "not_started"
    assert result["status_summary"]["blocked_reason"] == "gpu_free_memory_too_low"
    assert result["resource_blocker_snapshot"]["docker_container_stats_available"] is False
    assert result["resource_blocker_summary"]["docker_container_stats_reason"] == "docker_cli_unavailable"
    progress_events = [
        json.loads(line)["event"]
        for line in (tmp_path / "exec" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert progress_events == ["start", "gpu_preflight", "resource_snapshot_start", "resource_blocker_snapshot", "end"]
    summary_md = (tmp_path / "exec" / "summary.md").read_text(encoding="utf-8")
    assert "GPU preflight: blocked" in summary_md
    assert "free_mib=2048" in summary_md
    assert "Docker stats: unavailable (docker_cli_unavailable)" in summary_md
    assert "Result: not started" in summary_md


def test_run_stem_search_launch_manifest_blocks_on_recipe_sha_mismatch(tmp_path: Path) -> None:
    from tools.build_stem_search_launch_manifest import build_stem_search_launch_manifest
    from tools.run_stem_search_launch_manifest import run_stem_search_launch_manifest

    recipe_path = tmp_path / "recipe_after_policy.json"
    recipe_path.write_text(json.dumps({"schema": "stem_search_recipe.v2", "candidates": []}), encoding="utf-8")
    build_stem_search_launch_manifest(
        output_dir=tmp_path / "launch",
        mix_path=Path("/repo/testing_stuff_music/source.mp3"),
        recipe_path=recipe_path,
        run_output_dir=Path("/repo/artifacts/stem_eval/t999_run"),
        require_gpu=False,
        gpu_preflight={
            "schema": "stem_search_cuda_preflight.v1",
            "require_gpu": False,
            "min_free_mib": 0,
            "status": "not_required",
            "reason": "",
        },
    )
    recipe_path.write_text(json.dumps({"schema": "stem_search_recipe.v2", "candidates": [{"candidate_id": "c999"}]}), encoding="utf-8")

    result = run_stem_search_launch_manifest(
        launch_manifest_path=tmp_path / "launch" / "manifest.json",
        output_dir=tmp_path / "exec",
        command_runner=lambda command, timeout_sec: (_ for _ in ()).throw(AssertionError("must not run")),
        no_artifact_rag=True,
    )

    assert result["execution_status"] == "blocked_recipe_sha_mismatch"
    assert result["recipe_sha256_verified"] is False
    assert result["run_command_started"] is False
    assert result["delivery_allowed"] is False


def test_run_stem_search_launch_manifest_waits_until_gpu_ready(tmp_path: Path) -> None:
    from tools.build_stem_search_launch_manifest import build_stem_search_launch_manifest
    from tools.run_stem_search_launch_manifest import run_stem_search_launch_manifest

    recipe_path = tmp_path / "recipe_after_policy.json"
    recipe_path.write_text(json.dumps({"schema": "stem_search_recipe.v2", "candidates": []}), encoding="utf-8")
    build_stem_search_launch_manifest(
        output_dir=tmp_path / "launch",
        mix_path=Path("/repo/testing_stuff_music/source.mp3"),
        recipe_path=recipe_path,
        run_output_dir=Path("/repo/artifacts/stem_eval/t999_run"),
        wait_for_gpu_free_sec=5.0,
        gpu_preflight_poll_sec=1.0,
        gpu_preflight={
            "schema": "stem_search_cuda_preflight.v1",
            "require_gpu": True,
            "min_free_mib": 4096,
            "status": "blocked",
            "reason": "gpu_free_memory_too_low",
            "free_mib": 1024,
            "total_mib": 12282,
            "used_mib": 11258,
        },
    )
    preflights = [
        {
            "schema": "stem_search_cuda_preflight.v1",
            "require_gpu": True,
            "min_free_mib": 4096,
            "status": "blocked",
            "reason": "gpu_free_memory_too_low",
            "free_mib": 2048,
            "total_mib": 12282,
            "used_mib": 10234,
        },
        {
            "schema": "stem_search_cuda_preflight.v1",
            "require_gpu": True,
            "min_free_mib": 4096,
            "status": "passed",
            "reason": "",
            "free_mib": 8192,
            "total_mib": 12282,
            "used_mib": 4090,
        },
    ]
    calls: list[list[str]] = []
    sleeps: list[float] = []

    result = run_stem_search_launch_manifest(
        launch_manifest_path=tmp_path / "launch" / "manifest.json",
        output_dir=tmp_path / "exec",
        command_runner=lambda command, timeout_sec: calls.append(command),
        gpu_preflight_collector=lambda require_gpu, min_free_mib: preflights.pop(0),
        sleep_fn=lambda seconds: sleeps.append(seconds),
        no_artifact_rag=True,
    )

    assert result["execution_status"] == "command_completed"
    assert result["run_command_started"] is True
    assert result["recipe_sha256_verified"] is True
    assert len(calls) == 1
    assert sleeps == [1.0]
    assert result["gpu_preflight_summary"]["status"] == "passed"
    assert result["status_summary"]["result_status"] == "command_completed"
    assert result["status_summary"]["run_command_started"] is True
    progress_events = [
        json.loads(line)["event"]
        for line in (tmp_path / "exec" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert progress_events == [
        "start",
        "gpu_preflight_wait",
        "gpu_preflight",
        "resource_snapshot_start",
        "resource_blocker_snapshot",
        "command_start",
        "end",
    ]


def test_run_stem_search_launch_manifest_wait_override_extends_saved_launch(tmp_path: Path) -> None:
    from tools.build_stem_search_launch_manifest import build_stem_search_launch_manifest
    from tools.run_stem_search_launch_manifest import run_stem_search_launch_manifest

    recipe_path = tmp_path / "recipe_after_policy.json"
    recipe_path.write_text(json.dumps({"schema": "stem_search_recipe.v2", "candidates": []}), encoding="utf-8")
    build_stem_search_launch_manifest(
        output_dir=tmp_path / "launch",
        mix_path=Path("/repo/testing_stuff_music/source.mp3"),
        recipe_path=recipe_path,
        run_output_dir=Path("/repo/artifacts/stem_eval/t999_run"),
        wait_for_gpu_free_sec=0.0,
        gpu_preflight_poll_sec=1.0,
        gpu_preflight={
            "schema": "stem_search_cuda_preflight.v1",
            "require_gpu": True,
            "min_free_mib": 4096,
            "status": "blocked",
            "reason": "gpu_free_memory_too_low",
            "free_mib": 1024,
            "total_mib": 12282,
            "used_mib": 11258,
        },
    )
    preflights = [
        {
            "schema": "stem_search_cuda_preflight.v1",
            "require_gpu": True,
            "min_free_mib": 4096,
            "status": "blocked",
            "reason": "gpu_free_memory_too_low",
            "free_mib": 2048,
            "total_mib": 12282,
            "used_mib": 10234,
        },
        {
            "schema": "stem_search_cuda_preflight.v1",
            "require_gpu": True,
            "min_free_mib": 4096,
            "status": "passed",
            "reason": "",
            "free_mib": 8192,
            "total_mib": 12282,
            "used_mib": 4090,
        },
    ]
    calls: list[list[str]] = []
    sleeps: list[float] = []

    result = run_stem_search_launch_manifest(
        launch_manifest_path=tmp_path / "launch" / "manifest.json",
        output_dir=tmp_path / "exec",
        command_runner=lambda command, timeout_sec: calls.append(command),
        gpu_preflight_collector=lambda require_gpu, min_free_mib: preflights.pop(0),
        resource_snapshot_collector=lambda: {
            "schema": "stem_search_resource_blocker_snapshot.v1",
            "docker_container_stats_available": True,
            "docker_container_stats_reason": "",
            "top_container_memory": [],
            "gpu_processes": [],
        },
        sleep_fn=lambda seconds: sleeps.append(seconds),
        wait_for_gpu_free_sec_override=30.0,
        gpu_preflight_poll_sec_override=2.5,
        no_artifact_rag=True,
    )

    assert result["execution_status"] == "command_completed"
    assert result["run_command_started"] is True
    assert len(calls) == 1
    assert sleeps == [2.5]


def test_build_audio_iteration_analysis_ranks_current_next_action(tmp_path: Path) -> None:
    from tools.build_audio_iteration_analysis import build_audio_iteration_analysis

    policy_manifest = tmp_path / "policy.json"
    policy_manifest.write_text(
        json.dumps(
            {
                "schema": "root_cause_route_policy_inspection.v1",
                "delivery_allowed": False,
                "presentation_status": "diagnostic_policy_only",
                "repair_option_count_before_policy": 220,
                "remaining_repair_option_count": 110,
                "active_blocked_repair_option_count": 0,
                "root_cause_route_policy": {"skipped_repair_option_count": 110},
            }
        ),
        encoding="utf-8",
    )
    launch_manifest = tmp_path / "launch.json"
    launch_manifest.write_text(
        json.dumps(
            {
                "schema": "stem_search_launch_manifest.v1",
                "delivery_allowed": False,
                "presentation_status": "diagnostic_launch_only",
                "recipe_path": "/repo/artifacts/stem_eval/t353_policy_ready/recipe_after_policy.json",
                "run_output_dir": "/repo/artifacts/stem_eval/t355_full_filtered_replay",
                "launch_status": "blocked_gpu_preflight",
                "gpu_preflight": {"status": "blocked", "reason": "gpu_free_memory_too_low", "free_mib": 1488},
            }
        ),
        encoding="utf-8",
    )
    execution_manifest = tmp_path / "exec.json"
    execution_manifest.write_text(
        json.dumps(
            {
                "schema": "stem_search_launch_execution.v1",
                "delivery_allowed": False,
                "presentation_status": "diagnostic_launch_execution_only",
                "execution_status": "blocked_gpu_preflight",
                "run_command_started": False,
                "recipe_sha256_verified": True,
                "gpu_preflight": {"status": "blocked", "reason": "gpu_free_memory_too_low", "free_mib": 1488},
            }
        ),
        encoding="utf-8",
    )

    manifest = build_audio_iteration_analysis(
        output_dir=tmp_path / "iter",
        current_task="НАПИШИ_ЗАДАЧУ_ЗДЕСЬ: get filtered full-track replay running after GPU preflight blocker",
        policy_manifest_path=policy_manifest,
        launch_manifest_path=launch_manifest,
        launch_execution_manifest_path=execution_manifest,
        no_artifact_rag=True,
    )

    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "diagnostic_iteration_analysis_only"
    assert manifest["current_task"].startswith("get filtered full-track replay")
    assert "НАПИШИ_ЗАДАЧУ_ЗДЕСЬ" not in manifest["current_task"]
    assert manifest["confirmed"]["active_blocked_repair_option_count"] == 0
    assert manifest["confirmed"]["run_command_started"] is False
    assert manifest["selected_top_action"] == "gpu_resource_arbitration"
    assert manifest["rejected_idea_ids"] == ["threshold_relaxation", "human_review", "deliver_blocked_output"]
    assert (tmp_path / "iter" / "ideas.jsonl").exists()
    assert (tmp_path / "iter" / "summary.md").exists()
    progress_events = [
        json.loads(line)["event"]
        for line in (tmp_path / "iter" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert progress_events == ["start", "end"]


def test_audio_iteration_analysis_moves_to_gpu_arbitration_after_waited_preflight_block(tmp_path: Path) -> None:
    from tools.build_audio_iteration_analysis import build_audio_iteration_analysis

    execution_manifest = tmp_path / "exec.json"
    execution_manifest.write_text(
        json.dumps(
            {
                "schema": "stem_search_launch_execution.v1",
                "delivery_allowed": False,
                "presentation_status": "diagnostic_launch_execution_only",
                "execution_status": "blocked_gpu_preflight",
                "run_command_started": False,
                "recipe_sha256_verified": True,
                "gpu_preflight_summary": {
                    "status": "blocked",
                    "reason": "gpu_free_memory_too_low",
                    "free_mib": 1421,
                    "min_free_mib": 4096,
                    "used_mib": 10861,
                },
                "resource_blocker_summary": {
                    "docker_container_stats_available": False,
                    "docker_container_stats_reason": "docker_cli_unavailable",
                    "gpu_process_count": 0,
                    "top_container_memory_names": [],
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_audio_iteration_analysis(
        output_dir=tmp_path / "iter",
        current_task="НАПИШИ_ЗАДАЧУ_ЗДЕСЬ: continue after waited GPU preflight blocker",
        launch_execution_manifest_path=execution_manifest,
        no_artifact_rag=True,
    )

    assert manifest["selected_top_action"] == "gpu_resource_arbitration"
    assert manifest["confirmed"]["resource_blocker_summary"]["docker_container_stats_reason"] == "docker_cli_unavailable"
    assert manifest["confirmed"]["gpu_preflight_summary"]["free_mib"] == 1421
    summary_md = (tmp_path / "iter" / "summary.md").read_text(encoding="utf-8")
    assert "Selected top action: `gpu_resource_arbitration`" in summary_md
    assert "GPU free MiB: `1421`" in summary_md


def test_gpu_resource_arbitration_plan_is_diagnostic_and_non_destructive(tmp_path: Path) -> None:
    from tools.build_gpu_resource_arbitration_plan import build_gpu_resource_arbitration_plan

    iteration_manifest = tmp_path / "iter.json"
    iteration_manifest.write_text(
        json.dumps(
            {
                "schema": "audio_iteration_analysis.v1",
                "selected_top_action": "gpu_resource_arbitration",
                "delivery_allowed": False,
                "presentation_status": "diagnostic_iteration_analysis_only",
            }
        ),
        encoding="utf-8",
    )
    execution_manifest = tmp_path / "exec.json"
    execution_manifest.write_text(
        json.dumps(
            {
                "schema": "stem_search_launch_execution.v1",
                "execution_status": "blocked_gpu_preflight",
                "run_command_started": False,
                "delivery_allowed": False,
                "presentation_status": "diagnostic_launch_execution_only",
                "gpu_preflight_summary": {
                    "status": "blocked",
                    "reason": "gpu_free_memory_too_low",
                    "free_mib": 1406,
                    "min_free_mib": 4096,
                    "used_mib": 10592,
                    "total_mib": 12282,
                },
            }
        ),
        encoding="utf-8",
    )
    host_snapshot = {
        "schema": "stem_search_resource_blocker_snapshot.v1",
        "source": "test_host_snapshot",
        "docker_container_stats_available": True,
        "docker_container_stats_reason": "",
        "top_container_memory": [
            {
                "name": "trading-vllm-server",
                "cpu_percent": "2.32%",
                "mem_usage": "2.901GiB / 23.46GiB",
                "mem_percent": "12.36%",
            },
            {
                "name": "music-postgres",
                "cpu_percent": "0.00%",
                "mem_usage": "26.58MiB / 23.46GiB",
                "mem_percent": "0.11%",
            },
        ],
        "gpu_processes": [],
    }

    manifest = build_gpu_resource_arbitration_plan(
        output_dir=tmp_path / "arb",
        iteration_manifest_path=iteration_manifest,
        launch_execution_manifest_path=execution_manifest,
        resource_snapshot=host_snapshot,
        no_artifact_rag=True,
    )

    assert manifest["schema"] == "gpu_resource_arbitration_plan.v1"
    assert manifest["arbitration_status"] == "blocked_requires_external_gpu_release"
    assert manifest["recommended_action"] == "request_external_gpu_release_or_wait"
    assert manifest["destructive_action_started"] is False
    assert manifest["audio_replay_started"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "diagnostic_gpu_resource_arbitration_only"
    assert manifest["resource_blocker_summary"]["top_container_memory_names"][0] == "trading-vllm-server"
    assert "stop_container_without_permission" in manifest["forbidden_actions"]
    assert (tmp_path / "arb" / "actions.jsonl").exists()
    summary_md = (tmp_path / "arb" / "summary.md").read_text(encoding="utf-8")
    assert "Recommended action: `request_external_gpu_release_or_wait`" in summary_md
    assert "trading-vllm-server" in summary_md


def test_gpu_resource_arbitration_uses_execution_resource_snapshot_by_default(tmp_path: Path) -> None:
    from tools.build_gpu_resource_arbitration_plan import build_gpu_resource_arbitration_plan

    execution_manifest = tmp_path / "exec.json"
    execution_manifest.write_text(
        json.dumps(
            {
                "schema": "stem_search_launch_execution.v1",
                "execution_status": "blocked_gpu_preflight",
                "run_command_started": False,
                "gpu_preflight_summary": {
                    "status": "blocked",
                    "reason": "gpu_free_memory_too_low",
                    "free_mib": 1405,
                    "min_free_mib": 4096,
                    "used_mib": 10877,
                    "total_mib": 12282,
                },
                "resource_blocker_snapshot": {
                    "schema": "stem_search_resource_blocker_snapshot.v1",
                    "docker_container_stats_available": True,
                    "docker_container_stats_reason": "",
                    "docker_container_stats_source": "docker_engine_socket",
                    "top_container_memory": [
                        {
                            "name": "trading-vllm-server",
                            "cpu_percent": "",
                            "mem_usage": "3.000GiB / 24.000GiB",
                            "mem_percent": "12.50%",
                        }
                    ],
                    "gpu_processes": [],
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_gpu_resource_arbitration_plan(
        output_dir=tmp_path / "arb",
        launch_execution_manifest_path=execution_manifest,
        no_artifact_rag=True,
    )

    assert manifest["resource_blocker_summary"]["docker_container_stats_available"] is True
    assert manifest["resource_blocker_summary"]["top_container_memory_names"] == ["trading-vllm-server"]
    assert "trading-vllm-server" in (tmp_path / "arb" / "summary.md").read_text(encoding="utf-8")


def test_host_resource_snapshot_records_container_offenders_without_actions(tmp_path: Path) -> None:
    from tools.build_gpu_resource_arbitration_plan import build_gpu_resource_arbitration_plan
    from tools.build_host_resource_snapshot import build_host_resource_snapshot

    def fake_runner(command: list[str], timeout_sec: float) -> dict[str, object]:
        joined = " ".join(command)
        if "query-gpu" in joined:
            return {
                "available": True,
                "reason": "",
                "returncode": 0,
                "stdout": "NVIDIA GeForce RTX 4080 Laptop GPU, 12282, 10593, 1405, 99\n",
                "stderr": "",
            }
        if "query-compute-apps" in joined:
            return {
                "available": True,
                "reason": "",
                "returncode": 0,
                "stdout": "41720, Codex.exe, [N/A]\n",
                "stderr": "",
            }
        if command[:3] == ["docker", "stats", "--no-stream"]:
            return {
                "available": True,
                "reason": "",
                "returncode": 0,
                "stdout": (
                    "trading-vllm-server|109.06%|2.901GiB / 23.46GiB|12.36%\n"
                    "trading-view-llm-llama-web-proxy-1|0.28%|1.707GiB / 23.46GiB|7.28%\n"
                ),
                "stderr": "",
            }
        if command[:2] == ["docker", "ps"]:
            return {
                "available": True,
                "reason": "",
                "returncode": 0,
                "stdout": (
                    "trading-vllm-server|vllm/vllm-openai:latest|Up 4 hours (healthy)\n"
                    "trading-view-llm-llama-web-proxy-1|trading-view-llm-llama-web-proxy-portable:latest|Up 57 seconds\n"
                ),
                "stderr": "",
            }
        raise AssertionError(command)

    manifest = build_host_resource_snapshot(
        output_dir=tmp_path / "host",
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert manifest["schema"] == "host_resource_snapshot.v1"
    assert manifest["diagnostic_only"] is True
    assert manifest["execution_mode"] == "host_observability_only"
    assert manifest["destructive_action_started"] is False
    assert manifest["audio_replay_started"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "diagnostic_host_resource_snapshot_only"
    assert manifest["gpu_preflight_summary"]["free_mib"] == 1405
    assert manifest["resource_blocker_summary"]["top_container_memory_names"][0] == "trading-vllm-server"
    assert (tmp_path / "host" / "resource.json").exists()
    assert (tmp_path / "host" / "progress.jsonl").exists()

    arb = build_gpu_resource_arbitration_plan(
        output_dir=tmp_path / "arb_from_host",
        resource_snapshot=manifest["resource_blocker_snapshot"],
        no_artifact_rag=True,
    )
    assert arb["resource_blocker_summary"]["top_container_memory_names"][0] == "trading-vllm-server"


def test_host_resource_snapshot_import_does_not_require_audio_runtime_dependencies() -> None:
    script = """
import builtins
real_import = builtins.__import__
blocked = {"torch", "soundfile", "numpy", "tools.run_fully_automatic_stem_search"}
def guard(name, globals=None, locals=None, fromlist=(), level=0):
    if name in blocked:
        raise AssertionError(f"blocked heavy import: {name}")
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = guard
import tools.build_host_resource_snapshot  # noqa: F401
"""
    completed = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_route_policy_shadow_mode_writes_scores_without_reordering(tmp_path: Path) -> None:
    from tools.stem_route_policy_common import ACTION_PRIORITIZE_ROUTE
    from tools.stem_route_policy_model import save_stem_route_policy_checkpoint, train_stem_route_policy_from_rows

    row = {
        "schema": "stem_route_policy_row.v1",
        "row_id": "p000001",
        "source_run": "run.json",
        "window_id": "w000",
        "candidate_id": "c002",
        "route_signature": "stem_lane:better",
        "lane_routes": {"stem_lane": ["better"]},
        "issue_type": "bad_reconstruction",
        "root_cause": "routing_gap",
        "affected_stem": "drums",
        "gate_verdicts": {},
        "forensic_features": {},
        "damage_features": {},
        "reconstruction_features": {},
        "current_action": "block_unknown",
        "oracle_action": ACTION_PRIORITIZE_ROUTE,
        "oracle_label_source": "counterfactual_replay",
        "usable_for_training": True,
        "usable_for_promotion": False,
        "is_ground_truth": False,
        "label_type": "blocked_candidate_mining",
    }
    checkpoint = save_stem_route_policy_checkpoint(train_stem_route_policy_from_rows([row]), tmp_path / "model.pt")
    model = stem_search._load_learned_route_policy_checkpoint(checkpoint)
    recipe = {
        "schema": "stem_search_recipe.v2",
        "candidates": [
            {"candidate_id": "c001", "route_signature": "stem_lane:worse"},
            {"candidate_id": "c002", "route_signature": "stem_lane:better"},
        ],
    }

    updated = stem_search._apply_learned_route_policy_to_recipe(recipe, route_policy_model=model, route_policy_mode="shadow")
    manifest = build_candidate_manifest("c002", tmp_path / "c002")
    manifest.update({"route_signature": "stem_lane:better", "verdict": "blocked_bad_reconstruction", "delivery_allowed": False})
    annotated = stem_search._annotate_candidate_with_learned_route_policy(
        manifest,
        route_policy_model=model,
        route_policy_mode="shadow",
    )

    assert [candidate["candidate_id"] for candidate in updated["candidates"]] == ["c001", "c002"]
    assert annotated["route_policy_action"] == ACTION_PRIORITIZE_ROUTE
    assert annotated["route_policy_shadow_only"] is True
    assert annotated["delivery_allowed"] is False


def test_route_policy_rank_mode_reorders_but_cannot_accept(tmp_path: Path) -> None:
    from tools.stem_route_policy_common import ACTION_PRIORITIZE_ROUTE
    from tools.stem_route_policy_model import save_stem_route_policy_checkpoint, train_stem_route_policy_from_rows

    row = {
        "schema": "stem_route_policy_row.v1",
        "row_id": "p000001",
        "source_run": "run.json",
        "window_id": "w000",
        "candidate_id": "c002",
        "route_signature": "stem_lane:better",
        "lane_routes": {"stem_lane": ["better"]},
        "issue_type": "bad_reconstruction",
        "root_cause": "routing_gap",
        "affected_stem": "drums",
        "gate_verdicts": {},
        "forensic_features": {},
        "damage_features": {},
        "reconstruction_features": {},
        "current_action": "block_unknown",
        "oracle_action": ACTION_PRIORITIZE_ROUTE,
        "oracle_label_source": "counterfactual_replay",
        "usable_for_training": True,
        "usable_for_promotion": False,
        "is_ground_truth": False,
        "label_type": "blocked_candidate_mining",
    }
    checkpoint = save_stem_route_policy_checkpoint(train_stem_route_policy_from_rows([row]), tmp_path / "model.pt")
    model = stem_search._load_learned_route_policy_checkpoint(checkpoint)
    recipe = {
        "schema": "stem_search_recipe.v2",
        "candidates": [
            {"candidate_id": "c001", "route_signature": "stem_lane:worse"},
            {"candidate_id": "c002", "route_signature": "stem_lane:better"},
        ],
    }

    updated = stem_search._apply_learned_route_policy_to_recipe(recipe, route_policy_model=model, route_policy_mode="rank")
    manifest = build_candidate_manifest("c002", tmp_path / "c002")
    manifest.update({"route_signature": "stem_lane:better", "verdict": "blocked_unknown", "delivery_allowed": False})
    annotated = stem_search._annotate_candidate_with_learned_route_policy(
        manifest,
        route_policy_model=model,
        route_policy_mode="rank",
    )

    assert [candidate["candidate_id"] for candidate in updated["candidates"]] == ["c002", "c001"]
    assert annotated["route_policy_action"] == ACTION_PRIORITIZE_ROUTE
    assert annotated["delivery_allowed"] is False
    assert annotated["presentation_status"] == "blocked_from_listen_pack"


def test_best_bad_candidate_is_not_accepted() -> None:
    result = select_accepted_candidate(
        [
            {"candidate_id": "c001", "verdict": "blocked_leak", "score": 0.9},
            {"candidate_id": "c002", "verdict": "blocked_damage", "score": 0.8},
        ]
    )

    assert result["verdict"] == "blocked_no_clean_candidate"
    assert result["accepted_candidate_id"] == ""


def test_run_accept_requires_every_window_to_have_clean_candidate() -> None:
    result = select_accepted_candidate(
        [
            {"candidate_id": "c001", "window_id": "w000", "verdict": "blocked_leak"},
            {"candidate_id": "c001", "window_id": "w001", "verdict": "auto_accept"},
        ],
        required_window_ids=["w000", "w001"],
    )

    assert result["verdict"] == "blocked_no_clean_candidate"
    assert result["delivery_allowed"] is False
    assert result["accepted_candidate_id"] == ""
    assert result["blocked_window_ids"] == ["w000"]


def test_no_artifact_rag_write_in_scratch_search(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="artifact_rag"):
        validate_no_artifact_rag_output(Path("D:/AIMusicProject/artifact_rag/ar47"))

    mix = tmp_path / "mix.wav"
    mix.write_bytes(b"not-a-real-audio-file")
    out = tmp_path / "scratch" / "t001"

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=out,
        window_sec=6.0,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        allow_default_recipe=False,
    )

    assert result["verdict"] == "blocked_no_clean_candidate"
    assert result["delivery_status"] == "blocked"
    assert result["presentation_status"] == "blocked_from_listen_pack"
    assert result["listen_pack"] == {}
    assert (out / "run.json").exists()
    assert not (out / "listen").exists()
    assert result["artifact_rag_write_allowed"] is False
    assert not (out / "artifact_rag").exists()
    assert "human" not in json.dumps(result)
    assert result["training_first_root_cause_queue_manifest"].endswith("training/root_cause/manifest.json")
    assert (out / "training" / "root_cause" / "manifest.json").exists()
    assert (out / "training" / "root_cause" / "queue.jsonl").exists()


def test_split_audio_windows_merges_tiny_tail_into_previous_window(tmp_path: Path) -> None:
    mix = write_wav(tmp_path / "mix.wav", sine(220.0, seconds=1.025))

    windows = stem_search.split_audio_windows(mix, tmp_path / "split", window_sec=1.0)

    assert len(windows) == 1
    assert windows[0]["window_id"] == "w000"
    assert windows[0]["start_sec"] == 0.0
    assert windows[0]["end_sec"] == pytest.approx(1.025, abs=1.0 / SAMPLE_RATE)
    assert windows[0]["sample_count"] == int(round(1.025 * SAMPLE_RATE))
    assert windows[0]["short_tail_merged"] is True
    assert windows[0]["merged_tail_sec"] == pytest.approx(0.025, abs=1.0 / SAMPLE_RATE)


def test_default_from_scratch_recipe_uses_target_specific_real_routes() -> None:
    recipe = build_default_from_scratch_recipe(
        demucs_available=True,
        available_vocal_routes=("audio_separator_bs_roformer_vocals", "mdx23c_vocals"),
    )

    assert recipe["schema"] == "stem_search_recipe.v1"
    assert recipe["candidates"][0]["candidate_id"] == "c001"
    assert recipe["candidates"][0]["route_family"] == "target_specific_demucs_6s_audio_separator_bs_roformer_vocals"
    assert recipe["candidates"][0]["required_stems"] == list(STEMS)
    assert [route["role"] for route in recipe["candidates"][0]["target_specific_commands"]] == [
        "six_stems",
        "four_stem_non_vocal",
        "vocals",
    ]
    first_commands = recipe["candidates"][0]["target_specific_commands"]
    assert "run_demucs_six_stem_candidate.py" in " ".join(first_commands[0]["command_template"])
    assert "run_registered_non_vocal_candidate.py" in " ".join(first_commands[1]["command_template"])
    assert "export_audio_separator_vocal_full.py" in " ".join(first_commands[2]["command_template"])
    assert first_commands[1]["command_template"][first_commands[1]["command_template"].index("--model-id") + 1] == "scnet_xl_ihf_musdb18"
    assert [candidate["vocal_route_id"] for candidate in recipe["candidates"][:2]] == [
        "audio_separator_bs_roformer_vocals",
        "mdx23c_vocals",
    ]
    assert all(
        candidate["conservation_repair_stem"] == "other"
        for candidate in recipe["candidates"]
        if candidate["candidate_id"] != "c010"
    )
    assert all(candidate["max_conservation_repair_ratio"] == 0.30 for candidate in recipe["candidates"])
    assert all("contamination_checks" not in candidate for candidate in recipe["candidates"])
    assert len(recipe["candidates"]) == 10
    assert any(candidate.get("minus_vocal_debleed", {}).get("strength") == 1.0 for candidate in recipe["candidates"])
    assert any(candidate.get("vocal_projection_debleed", {}).get("strength") == 1.0 for candidate in recipe["candidates"])
    assert any(
        "run_registered_six_stem_candidate.py" in " ".join(candidate.get("command_template") or [])
        for candidate in recipe["candidates"]
    )
    assert any(
        route.get("role") == "four_stem_non_vocal"
        for candidate in recipe["candidates"]
        for route in candidate.get("target_specific_commands", [])
    )
    c009 = next(candidate for candidate in recipe["candidates"] if candidate["candidate_id"] == "c009")
    c010 = next(candidate for candidate in recipe["candidates"] if candidate["candidate_id"] == "c010")
    assert c009["vocal_route_id"] == "audio_separator_bs_roformer_vocals"
    assert c010["route_family"] == "registered_six_stem_hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first"
    assert c010["registered_model_id"] == "hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first"
    assert c010["conservation_repair_stem"] == "guitar"
    assert c010["minus_vocal_debleed"]["passes"] == 2
    assert c010["post_minus_vocal_debleed"]["passes"] == 1
    assert c010["stem_merges"] == [
        {
            "source_stem": "other",
            "target_stem": "guitar",
            "mode": "move",
            "reason": "aggressive_melody_first_harmonic_bus",
        }
    ]
    assert (
        c010["command_template"][c010["command_template"].index("--model-id") + 1]
        == "hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first"
    )
    assert any(
        "--residual-correction-stem" in route["command_template"]
        for candidate in recipe["candidates"]
        for route in candidate.get("target_specific_commands", [])
    )


def test_default_from_scratch_recipe_records_contamination_model_manifest(tmp_path: Path) -> None:
    model_manifest = tmp_path / "cycle001" / "model" / "manifest.json"
    model_manifest.parent.mkdir(parents=True)
    model_manifest.write_text("{}", encoding="utf-8")

    recipe = build_default_from_scratch_recipe(
        demucs_available=True,
        available_vocal_routes=("audio_separator_bs_roformer_vocals",),
        contamination_model_manifest=model_manifest,
    )

    assert recipe["contamination_model_manifest"] == str(model_manifest)


def test_default_contamination_model_manifest_discovery_uses_latest_cycle(tmp_path: Path) -> None:
    old_manifest = tmp_path / "stem_eval" / "cycle002" / "model" / "manifest.json"
    new_manifest = tmp_path / "stem_eval" / "cycle010" / "model" / "manifest.json"
    old_manifest.parent.mkdir(parents=True)
    new_manifest.parent.mkdir(parents=True)
    old_manifest.write_text("{}", encoding="utf-8")
    new_manifest.write_text("{}", encoding="utf-8")

    discovered = _discover_default_contamination_model_manifest((tmp_path / "stem_eval",))

    assert discovered == new_manifest


def test_default_contamination_model_manifest_searches_project_artifacts_mount() -> None:
    assert stem_search.DEFAULT_CONTAMINATION_MODEL_SEARCH_ROOTS[0] == Path("/project_artifacts/stem_eval")
    assert Path("/repo/artifacts/stem_eval") in stem_search.DEFAULT_CONTAMINATION_MODEL_SEARCH_ROOTS
    assert Path("../artifacts/stem_eval") in stem_search.DEFAULT_CONTAMINATION_MODEL_SEARCH_ROOTS


def test_default_from_scratch_recipe_blocks_without_specialized_vocal_route() -> None:
    recipe = build_default_from_scratch_recipe(demucs_available=True, available_vocal_routes=())

    assert recipe["available"] is False
    assert recipe["unavailable_reason"] == "missing_target_specific_vocal_route"
    assert recipe["candidates"] == []


def test_model_split_v2_recipe_builds_separate_lanes() -> None:
    recipe = build_model_split_v2_recipe(
        demucs_available=True,
        available_vocal_routes=("audio_separator_bs_roformer_vocals", "mdx23c_vocals"),
    )

    assert recipe["schema"] == "stem_search_recipe.v2"
    assert recipe["profile"] == "model_split_v2"
    assert recipe["target_specific_routes"]["vocal_lane"] == [
        "audio_separator_bs_roformer_vocals",
        "mdx23c_vocals",
        "melband_roformer_vocals",
    ]
    assert recipe["target_specific_routes"]["minus_lane"] == [
        "becruily_instrumental_vocals",
        "audio_separator_bs_roformer_vocals:instrumental",
        "mdx23c_vocals:instrumental",
        "htdemucs_ft:instrumental",
    ]
    assert recipe["target_specific_routes"]["stem_lane"] == [
        "htdemucs_6s",
        "hybrid_htdemucs_ft_bs_roformer_6s",
        "hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first",
    ]
    assert recipe["target_specific_routes"]["bass_support_lane"] == [
        "scnet_xl_ihf_musdb18:bass",
    ]
    assert recipe["target_specific_routes"]["non_vocal_support_lane"] == [
        "scnet_xl_ihf_musdb18:drums_bass_other",
    ]
    assert recipe["model_split_composite_max_candidates"] == 864
    first = recipe["candidates"][0]
    assert first["minus_source"] == "minus_lane"
    assert first["minus_rebuilt_from_non_vocal_stems"] is False
    assert first["delivery_pair_reconstruction"] == {
        "enabled": True,
        "max_delivery_pair_residual_ratio": 0.06,
        "six_stem_diagnostic_residual_ratio": 0.01,
    }


def test_model_split_v2_prefers_resident_capable_separator_routes() -> None:
    recipe = build_model_split_v2_recipe(
        demucs_available=True,
        available_vocal_routes=("audio_separator_bs_roformer_vocals",),
    )

    resident_capable_items = [
        item
        for candidate in recipe["candidates"]
        for item in candidate["target_specific_commands"]
        if str(item["route_id"]).split(":", 1)[0]
        in {"audio_separator_bs_roformer_vocals", "becruily_instrumental_vocals", "melband_roformer_vocals"}
    ]

    assert resident_capable_items
    assert {item["route_id"] for item in resident_capable_items} >= {
        "audio_separator_bs_roformer_vocals",
        "audio_separator_bs_roformer_vocals:instrumental",
        "becruily_instrumental_vocals",
    }
    assert all(item.get("prefer_resident") is True for item in resident_capable_items)
    first = recipe["candidates"][0]
    assert first["model_split_lanes"] == {
        "vocals": "vocal_lane",
        "minus": "minus_lane",
        "stems": "stem_lane",
        "bass_support": "bass_support_lane",
        "non_vocal_support": "non_vocal_support_lane",
    }
    assert [option["repair_id"] for option in first["model_split_composite_repair_options"]] == [
        "none",
        "vp100",
        "bs250",
        "bs400",
        "pg08",
        "pr16",
        "bs400_pr16",
        "vp100_bs400",
        "bs400_pg08",
        "vp100_pg08",
        "vp100_bs400_pg08",
    ]
    assert [route["role"] for route in first["target_specific_commands"]] == [
        "six_stems",
        "vocals",
        "minus",
        "non_vocal_support",
    ]
    non_vocal_support = first["target_specific_commands"][-1]
    assert non_vocal_support["lane"] == "non_vocal_support_lane"
    assert "run_registered_non_vocal_candidate.py" in " ".join(non_vocal_support["command_template"])
    assert non_vocal_support["command_template"][non_vocal_support["command_template"].index("--model-id") + 1] == "scnet_xl_ihf_musdb18"


def test_model_split_v2_recipe_materializes_support_lanes_beyond_first_candidate() -> None:
    recipe = build_model_split_v2_recipe(
        demucs_available=True,
        available_vocal_routes=("audio_separator_bs_roformer_vocals", "mdx23c_vocals"),
    )
    support_roles_by_candidate = {
        candidate["candidate_id"]: {
            command["role"]
            for command in candidate["target_specific_commands"]
            if command["role"] in {"bass_support", "non_vocal_support"}
        }
        for candidate in recipe["candidates"]
    }

    assert any("bass_support" in roles for roles in support_roles_by_candidate.values())
    assert any(
        "non_vocal_support" in roles
        for candidate_id, roles in support_roles_by_candidate.items()
        if candidate_id != "c001"
    )


def test_model_split_v2_recipe_can_inject_teacher_mask_checkpoint_routes(tmp_path: Path) -> None:
    minus_checkpoint = tmp_path / "m" / "model.pt"
    bass_checkpoint = tmp_path / "b" / "model.pt"
    other_checkpoint = tmp_path / "o" / "model.pt"
    recipe = build_model_split_v2_recipe(
        demucs_available=True,
        teacher_mask_checkpoints=[
            {
                "route_id": "tm001_minus",
                "role": "minus",
                "stem": "minus",
                "checkpoint": str(minus_checkpoint),
            },
            {
                "route_id": "tm002_bass",
                "role": "bass_support",
                "stem": "bass",
                "checkpoint": str(bass_checkpoint),
            },
            {
                "route_id": "tm003_other",
                "role": "other_support",
                "stem": "other",
                "checkpoint": str(other_checkpoint),
            },
        ],
    )

    first = recipe["candidates"][0]
    teacher_routes = [
        route
        for route in first["target_specific_commands"]
        if route["lane"] == "teacher_mask_checkpoint_lane"
    ]

    assert [route["role"] for route in teacher_routes] == ["minus", "bass_support", "other_support"]
    assert recipe["target_specific_routes"]["teacher_mask_checkpoint_lane"] == [
        "tm001_minus",
        "tm002_bass",
        "tm003_other",
    ]
    assert first["minus_source"] == "minus_lane"
    assert first["minus_rebuilt_from_non_vocal_stems"] is False
    assert "teacher_mask_checkpoint" in first["route_family"]
    assert "apply_teacher_mask_checkpoint_candidate.py" in " ".join(teacher_routes[0]["command_template"])
    assert teacher_routes[0]["command_template"][teacher_routes[0]["command_template"].index("--checkpoint") + 1] == str(
        minus_checkpoint
    )
    assert teacher_routes[1]["command_template"][teacher_routes[1]["command_template"].index("--stem") + 1] == "bass"
    assert teacher_routes[2]["command_template"][teacher_routes[2]["command_template"].index("--stem") + 1] == "other"


def test_teacher_mask_checkpoint_spec_preserves_windows_path() -> None:
    spec = _parse_teacher_mask_checkpoint_spec("minus:minus:D:\\AIMusicProject\\artifacts\\model.pt", index=3)

    assert spec == {
        "route_id": "tm003_minus",
        "role": "minus",
        "stem": "minus",
        "checkpoint": "D:\\AIMusicProject\\artifacts\\model.pt",
    }


def test_stem_region_repair_checkpoint_spec_accepts_json_fields() -> None:
    spec = stem_search._parse_stem_region_repair_checkpoint_spec(
        json.dumps(
            {
                "repair_id": "vr001",
                "stem": "vocals",
                "issue_type": "instrument_leak",
                "checkpoint": "/repo/artifacts/model.pt",
                "forensic_source_condition_stem": "drums",
                "mix_preserve_companion_stem": "minus",
                "region": {"start_sec": 0.2, "end_sec": 1.35},
            }
        ),
        index=7,
    )

    assert spec["repair_id"] == "vr001"
    assert spec["stem"] == "vocals"
    assert spec["issue_type"] == "instrument_leak"
    assert spec["checkpoint"] == "/repo/artifacts/model.pt"
    assert spec["forensic_source_condition_stem"] == "drums"
    assert spec["mix_preserve_companion_stem"] == "minus"
    assert spec["region"] == {"start_sec": 0.2, "end_sec": 1.35}


def test_stem_region_repair_checkpoint_spec_accepts_json_file(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "stem": "vocals",
                "issue_type": "instrument_leak",
                "checkpoint": "/repo/artifacts/model.pt",
                "forensic_source_condition_stem": "drums",
            }
        ),
        encoding="utf-8",
    )

    spec = stem_search._parse_stem_region_repair_checkpoint_spec(f"@{spec_path}", index=2)

    assert spec["repair_id"] == "sr002"
    assert spec["stem"] == "vocals"
    assert spec["forensic_source_condition_stem"] == "drums"


def test_model_split_v2_recipe_can_inject_learned_region_repair_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model" / "model.pt"
    recipe = build_model_split_v2_recipe(
        demucs_available=True,
        stem_region_repair_checkpoints=[
            {
                "repair_id": "sr001",
                "stem": "bass",
                "issue_type": ISSUE_STEREO_COLLAPSE,
                "checkpoint": str(checkpoint),
            }
        ],
    )

    repairs = recipe["candidates"][0]["model_split_composite_repair_options"]

    assert repairs[0] == {"repair_id": "none", "type": "none"}
    assert repairs[1] == {
        "repair_id": "sr001",
        "type": "stem_region_repair",
        "stem": "bass",
        "issue_type": ISSUE_STEREO_COLLAPSE,
        "checkpoint": str(checkpoint),
    }
    assert repairs[2] == {
        "repair_id": "sr001_vp100_pr16",
        "type": "stem_region_repair+vocal_projection_debleed+delivery_pair_residual",
        "stem_region_repair": {
            "stem": "bass",
            "issue_type": ISSUE_STEREO_COLLAPSE,
            "checkpoint": str(checkpoint),
        },
        "vocal_projection_debleed": {
            "strength": 1.0,
            "target_stems": ["drums", "bass", "guitar", "piano", "other"],
        },
        "delivery_pair_residual": {
            "target_stem": "minus",
            "max_residual_ratio": 0.16,
        },
    }
    assert repairs[3] == {
        "repair_id": "sr001_vp100",
        "type": "stem_region_repair+vocal_projection_debleed",
        "stem_region_repair": {
            "stem": "bass",
            "issue_type": ISSUE_STEREO_COLLAPSE,
            "checkpoint": str(checkpoint),
        },
        "vocal_projection_debleed": {
            "strength": 1.0,
            "target_stems": ["drums", "bass", "guitar", "piano", "other"],
        },
    }


def test_model_split_v2_recipe_can_inject_learned_region_repair_sequence(tmp_path: Path) -> None:
    checkpoints = [
        tmp_path / "m1" / "model.pt",
        tmp_path / "m2" / "model.pt",
        tmp_path / "m3" / "model.pt",
    ]
    recipe = build_model_split_v2_recipe(
        demucs_available=True,
        stem_region_repair_checkpoints=[
            {"repair_id": "sr001", "stem": "bass", "issue_type": "vocal_leak", "checkpoint": str(checkpoints[0])},
            {"repair_id": "sr002", "stem": "minus", "issue_type": "vocal_leak", "checkpoint": str(checkpoints[1])},
            {"repair_id": "sr003", "stem": "other", "issue_type": "instrument_leak", "checkpoint": str(checkpoints[2])},
        ],
    )

    repairs = recipe["candidates"][0]["model_split_composite_repair_options"]
    sequence_plain = next(repair for repair in repairs if repair["repair_id"] == "srseq")
    sequence_bass = next(repair for repair in repairs if repair["repair_id"] == "srseq_bs400")
    sequence_vp100 = next(repair for repair in repairs if repair["repair_id"] == "srseq_vp100")
    sequence_vp100_bass = next(repair for repair in repairs if repair["repair_id"] == "srseq_vp100_bs400")
    sequence = next(repair for repair in repairs if repair["repair_id"] == "srseq_vp100_pr16")

    assert sequence_plain["type"] == "stem_region_repair"
    assert "delivery_pair_residual" not in sequence_plain
    assert sequence_plain["stem_region_repairs"] == [
        {"stem": "bass", "issue_type": "vocal_leak", "checkpoint": str(checkpoints[0])},
        {
            "stem": "minus",
            "issue_type": "vocal_leak",
            "checkpoint": str(checkpoints[1]),
            "mix_preserve_companion_stem": "vocals",
        },
        {"stem": "other", "issue_type": "instrument_leak", "checkpoint": str(checkpoints[2])},
    ]
    assert sequence_vp100["type"] == "stem_region_repair+vocal_projection_debleed"
    assert "delivery_pair_residual" not in sequence_vp100
    assert sequence_vp100["vocal_projection_debleed"]["target_stems"] == ["drums", "bass", "guitar", "piano", "other"]
    assert sequence_bass["type"] == "stem_region_repair+bass_stereo_width"
    assert sequence_bass["bass_stereo_width"]["side_gain"] == 4.0
    assert sequence_vp100_bass["type"] == "stem_region_repair+vocal_projection_debleed+bass_stereo_width"
    assert sequence_vp100_bass["bass_stereo_width"]["side_gain"] == 4.0
    assert sequence["type"] == "stem_region_repair+vocal_projection_debleed+delivery_pair_residual"
    assert sequence["stem_region_repairs"] == [
        {"stem": "bass", "issue_type": "vocal_leak", "checkpoint": str(checkpoints[0])},
        {
            "stem": "minus",
            "issue_type": "vocal_leak",
            "checkpoint": str(checkpoints[1]),
            "mix_preserve_companion_stem": "vocals",
        },
        {"stem": "other", "issue_type": "instrument_leak", "checkpoint": str(checkpoints[2])},
    ]
    assert sequence["vocal_projection_debleed"]["target_stems"] == ["drums", "bass", "guitar", "piano", "other"]
    assert sequence["delivery_pair_residual"]["target_stem"] == "minus"


def test_model_split_v2_region_repair_preserves_forensic_source_and_pair_compensation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "v" / "model.pt"
    recipe = build_model_split_v2_recipe(
        demucs_available=True,
        stem_region_repair_checkpoints=[
            {
                "repair_id": "vr001",
                "stem": "vocals",
                "issue_type": "instrument_leak",
                "checkpoint": str(checkpoint),
                "forensic_source_condition_stem": "drums",
                "mix_preserve_companion_stem": "minus",
                "region": {"start_sec": 0.2, "end_sec": 1.35},
            }
        ],
    )

    repairs = recipe["candidates"][0]["model_split_composite_repair_options"]
    repair = next(item for item in repairs if item["repair_id"] == "vr001")

    assert repair["stem"] == "vocals"
    assert repair["issue_type"] == "instrument_leak"
    assert repair["forensic_source_condition_stem"] == "drums"
    assert repair["mix_preserve_companion_stem"] == "minus"
    assert repair["region"] == {"start_sec": 0.2, "end_sec": 1.35}


def test_apply_learned_stem_region_repairs_runs_sequence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    applied: list[str] = []
    stem_paths = {
        "bass": str(tmp_path / "bass.flac"),
        "minus": str(tmp_path / "minus.flac"),
        "other": str(tmp_path / "other.flac"),
    }

    def fake_apply_learned_stem_region_repair(*, mix_path: Path, stem_paths: dict[str, str], config: dict[str, str]) -> dict:
        applied.append(f"{config['stem']}:{config['issue_type']}")
        return {
            "applied": True,
            "stem": config["stem"],
            "issue_type": config["issue_type"],
            "checkpoint": config["checkpoint"],
            "blockers": [],
        }

    monkeypatch.setattr(stem_search, "apply_learned_stem_region_repair", fake_apply_learned_stem_region_repair)

    result = stem_search.apply_learned_stem_region_repairs(
        mix_path=tmp_path / "mix.flac",
        stem_paths=stem_paths,
        configs=[
            {"stem": "bass", "issue_type": "vocal_leak", "checkpoint": "b.pt"},
            {"stem": "minus", "issue_type": "vocal_leak", "checkpoint": "m.pt"},
            {"stem": "other", "issue_type": "instrument_leak", "checkpoint": "o.pt"},
        ],
    )

    assert applied == ["bass:vocal_leak", "minus:vocal_leak", "other:instrument_leak"]
    assert result["applied"] is True
    assert result["blockers"] == []
    assert [item["stem"] for item in result["repairs"]] == ["bass", "minus", "other"]


def test_apply_learned_minus_vocal_leak_repair_can_preserve_vocal_minus_sum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = SAMPLE_RATE // 2
    t = np.arange(frames, dtype=np.float32) / float(SAMPLE_RATE)
    vocals = np.stack(
        [
            0.08 * np.sin(2.0 * np.pi * 720.0 * t),
            0.07 * np.sin(2.0 * np.pi * 725.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    body = np.stack(
        [
            0.18 * np.sin(2.0 * np.pi * 110.0 * t),
            0.17 * np.sin(2.0 * np.pi * 114.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    leak = np.stack(
        [
            0.025 * np.sin(2.0 * np.pi * 840.0 * t),
            0.025 * np.sin(2.0 * np.pi * 845.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    bad_minus = body + leak
    mix = vocals + bad_minus
    mix_path = write_wav(tmp_path / "mix.flac", mix)
    vocals_path = write_wav(tmp_path / "vocals.flac", vocals)
    minus_path = write_wav(tmp_path / "minus.flac", bad_minus)

    class FakeModel:
        pass

    def fake_load_region_repair_checkpoint(_path: Path):
        return FakeModel(), {"training_history": {"issue_type": "vocal_leak", "region": {"start_sec": 0.0, "end_sec": 0.5}}}

    def fake_apply_region_repair_model(**_kwargs):
        return body, {"schema": "fake_repair"}

    monkeypatch.setattr(stem_search, "load_region_repair_checkpoint", fake_load_region_repair_checkpoint)
    monkeypatch.setattr(stem_search, "apply_region_repair_model", fake_apply_region_repair_model)

    result = stem_search.apply_learned_stem_region_repair(
        mix_path=mix_path,
        stem_paths={"vocals": str(vocals_path), "minus": str(minus_path)},
        config={
            "stem": "minus",
            "issue_type": "vocal_leak",
            "checkpoint": str(tmp_path / "model.pt"),
            "mix_preserve_companion_stem": "vocals",
        },
    )

    vocals_after, _ = stem_search._read_audio(vocals_path)
    minus_after, _ = stem_search._read_audio(minus_path)

    assert result["applied"] is True
    assert result["mix_preserve_companion_stem"] == "vocals"
    assert np.max(np.abs((vocals_after + minus_after) - mix)) < 2.0e-4
    assert np.max(np.abs(minus_after - body)) < 2.0e-4


def test_apply_learned_other_vocal_transfer_preserves_delivery_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = SAMPLE_RATE // 2
    t = np.arange(frames, dtype=np.float32) / float(SAMPLE_RATE)
    vocals = np.stack(
        [
            0.08 * np.sin(2.0 * np.pi * 720.0 * t),
            0.07 * np.sin(2.0 * np.pi * 725.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    minus = np.stack(
        [
            0.18 * np.sin(2.0 * np.pi * 110.0 * t),
            0.17 * np.sin(2.0 * np.pi * 114.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    other_clean = np.stack(
        [
            0.025 * np.sin(2.0 * np.pi * 2200.0 * t),
            0.023 * np.sin(2.0 * np.pi * 2210.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    leak = np.stack(
        [
            0.02 * np.sin(2.0 * np.pi * 840.0 * t),
            0.02 * np.sin(2.0 * np.pi * 845.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    bad_other = other_clean + leak
    mix = vocals + minus
    mix_path = write_wav(tmp_path / "mix.flac", mix)
    vocals_path = write_wav(tmp_path / "vocals.flac", vocals)
    minus_path = write_wav(tmp_path / "minus.flac", minus)
    other_path = write_wav(tmp_path / "other.flac", bad_other)

    class FakeModel:
        pass

    def fake_load_region_repair_checkpoint(_path: Path):
        return FakeModel(), {
            "training_history": {
                "issue_type": "vocal_leak",
                "region": {"start_sec": 0.0, "end_sec": 0.5},
                "mix_preserve_companion_stem": "vocals",
            }
        }

    def fake_apply_region_repair_model(**_kwargs):
        return other_clean, {"schema": "fake_repair"}

    monkeypatch.setattr(stem_search, "load_region_repair_checkpoint", fake_load_region_repair_checkpoint)
    monkeypatch.setattr(stem_search, "apply_region_repair_model", fake_apply_region_repair_model)

    result = stem_search.apply_learned_stem_region_repair(
        mix_path=mix_path,
        stem_paths={"vocals": str(vocals_path), "minus": str(minus_path), "other": str(other_path)},
        config={
            "stem": "other",
            "issue_type": "vocal_leak",
            "checkpoint": str(tmp_path / "model.pt"),
        },
    )

    vocals_after, _ = stem_search._read_audio(vocals_path)
    minus_after, _ = stem_search._read_audio(minus_path)
    other_after, _ = stem_search._read_audio(other_path)

    assert result["applied"] is True
    assert result["mix_preserve_companion_stem"] == "vocals"
    assert result["delivery_pair_compensation_stem"] == "minus"
    assert np.max(np.abs((vocals_after + minus_after) - mix)) < 2.0e-4
    assert np.max(np.abs(other_after - other_clean)) < 2.0e-4


def test_vocal_instrument_leak_repair_does_not_inject_delta_into_minus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = SAMPLE_RATE // 2
    t = np.arange(frames, dtype=np.float32) / float(SAMPLE_RATE)
    clean_vocals = np.stack(
        [
            0.07 * np.sin(2.0 * np.pi * 620.0 * t),
            0.065 * np.sin(2.0 * np.pi * 625.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    drum_imprint = np.stack(
        [
            0.025 * np.sin(2.0 * np.pi * 2600.0 * t),
            0.025 * np.sin(2.0 * np.pi * 2630.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    bad_vocals = clean_vocals + drum_imprint
    drums = np.zeros_like(clean_vocals)
    minus = np.stack(
        [
            0.12 * np.sin(2.0 * np.pi * 120.0 * t),
            0.115 * np.sin(2.0 * np.pi * 124.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    mix = clean_vocals + minus
    mix_path = write_wav(tmp_path / "mix.flac", mix)
    vocals_path = write_wav(tmp_path / "vocals.flac", bad_vocals)
    drums_path = write_wav(tmp_path / "drums.flac", drums)
    minus_path = write_wav(tmp_path / "minus.flac", minus)

    class FakeModel:
        pass

    def fake_load_region_repair_checkpoint(_path: Path):
        return FakeModel(), {
            "training_history": {
                "issue_type": "instrument_leak",
                "region": {"start_sec": 0.0, "end_sec": 0.5},
                "mix_preserve_companion_stem": "drums",
            }
        }

    def fake_apply_region_repair_model(**_kwargs):
        return clean_vocals, {"schema": "fake_repair"}

    monkeypatch.setattr(stem_search, "load_region_repair_checkpoint", fake_load_region_repair_checkpoint)
    monkeypatch.setattr(stem_search, "apply_region_repair_model", fake_apply_region_repair_model)

    result = stem_search.apply_learned_stem_region_repair(
        mix_path=mix_path,
        stem_paths={"vocals": str(vocals_path), "drums": str(drums_path), "minus": str(minus_path)},
        config={
            "stem": "vocals",
            "issue_type": "instrument_leak",
            "checkpoint": str(tmp_path / "model.pt"),
        },
    )

    vocals_after, _ = stem_search._read_audio(vocals_path)
    drums_after, _ = stem_search._read_audio(drums_path)
    minus_after, _ = stem_search._read_audio(minus_path)

    assert result["applied"] is True
    assert result["mix_preserve_companion_stem"] == "drums"
    assert result["delivery_pair_compensation_stem"] == ""
    assert np.max(np.abs(vocals_after - clean_vocals)) < 2.0e-4
    assert np.max(np.abs(drums_after - drum_imprint)) < 2.0e-4
    assert np.max(np.abs(minus_after - minus)) < 2.0e-4


def test_vocal_instrument_leak_repair_preserves_delivery_pair_with_minus_companion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = SAMPLE_RATE // 2
    t = np.arange(frames, dtype=np.float32) / float(SAMPLE_RATE)
    clean_vocals = np.stack(
        [
            0.07 * np.sin(2.0 * np.pi * 620.0 * t),
            0.065 * np.sin(2.0 * np.pi * 625.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    drum_imprint = np.stack(
        [
            0.025 * np.sin(2.0 * np.pi * 2600.0 * t),
            0.025 * np.sin(2.0 * np.pi * 2630.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    bad_vocals = clean_vocals + drum_imprint
    drums = drum_imprint.copy()
    minus = np.stack(
        [
            0.12 * np.sin(2.0 * np.pi * 120.0 * t),
            0.115 * np.sin(2.0 * np.pi * 124.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    mix = clean_vocals + minus + drum_imprint
    mix_path = write_wav(tmp_path / "mix.flac", mix)
    vocals_path = write_wav(tmp_path / "vocals.flac", bad_vocals)
    drums_path = write_wav(tmp_path / "drums.flac", drums)
    minus_path = write_wav(tmp_path / "minus.flac", minus)

    class FakeModel:
        pass

    def fake_load_region_repair_checkpoint(_path: Path):
        return FakeModel(), {
            "training_history": {
                "issue_type": "instrument_leak",
                "region": {"start_sec": 0.0, "end_sec": 0.5},
                "mix_preserve_companion_stem": "minus",
                "forensic_source_condition_stem": "drums",
            }
        }

    def fake_apply_region_repair_model(**_kwargs):
        return clean_vocals, {"schema": "fake_repair"}

    monkeypatch.setattr(stem_search, "load_region_repair_checkpoint", fake_load_region_repair_checkpoint)
    monkeypatch.setattr(stem_search, "apply_region_repair_model", fake_apply_region_repair_model)

    result = stem_search.apply_learned_stem_region_repair(
        mix_path=mix_path,
        stem_paths={"vocals": str(vocals_path), "drums": str(drums_path), "minus": str(minus_path)},
        config={
            "stem": "vocals",
            "issue_type": "instrument_leak",
            "checkpoint": str(tmp_path / "model.pt"),
        },
    )

    vocals_after, _ = stem_search._read_audio(vocals_path)
    minus_after, _ = stem_search._read_audio(minus_path)

    assert result["applied"] is True
    assert result["mix_preserve_companion_stem"] == "minus"
    assert result["delivery_pair_compensation_stem"] == ""
    assert np.max(np.abs(vocals_after - clean_vocals)) < 2.0e-4
    assert np.max(np.abs(minus_after - (minus + drum_imprint))) < 2.0e-4
    assert np.max(np.abs((vocals_after + minus_after) - mix)) < 2.0e-4


def test_delivery_pair_residual_blocks_after_vocal_instrument_leak_repair() -> None:
    blocker = stem_search._delivery_pair_residual_blocker_after_stem_region_repair(
        {
            "applied": True,
            "stem": "vocals",
            "issue_type": "instrument_leak",
            "forensic_source_condition_stem": "drums",
        },
        {"target_stem": "minus"},
    )

    assert blocker == "delivery_pair_residual_after_vocal_instrument_leak_repair_blocked"


def test_model_split_v2_composite_prioritizes_learned_region_repair_before_dsp_repairs() -> None:
    pairs = stem_search._prioritized_model_split_support_repair_pairs(
        support_options=[
            {},
            {"bass": {"role": "bass_support"}},
            {"drums": {"role": "non_vocal_support"}, "bass": {"role": "non_vocal_support"}},
        ],
        repair_options=[
            {"repair_id": "none", "type": "none"},
            {"repair_id": "sr001", "type": "stem_region_repair"},
            {"repair_id": "pr16", "type": "delivery_pair_residual"},
            {"repair_id": "bs400_pr16", "type": "bass_stereo_width+delivery_pair_residual"},
        ],
    )

    ordered = [
        (repair["repair_id"], "bass" in support, any((route.get("role") == "non_vocal_support") for route in support.values()))
        for support, repair in pairs
    ]
    first_six = ordered[:6]

    assert ("sr001", True, False) in first_six
    assert ("sr001", True, True) in first_six
    assert ordered.index(("sr001", True, False)) < ordered.index(("pr16", True, False))


def test_model_split_v2_composite_reaches_learned_repair_before_none_support_budget_exhaustion() -> None:
    pairs = stem_search._prioritized_model_split_support_repair_pairs(
        support_options=[
            {},
            {"bass": {"role": "bass_support"}},
            {"other": {"role": "other_support"}},
            {"bass": {"role": "bass_support"}, "other": {"role": "other_support"}},
            {"drums": {"role": "non_vocal_support"}},
            {"bass": {"role": "bass_support"}, "drums": {"role": "non_vocal_support"}},
        ],
        repair_options=[
            {"repair_id": "none", "type": "none"},
            {"repair_id": "sr001", "type": "stem_region_repair"},
            {"repair_id": "bs400", "type": "bass_stereo_width"},
        ],
    )

    first_five = [(repair["repair_id"], sorted(support)) for support, repair in pairs[:5]]

    assert any(repair_id == "sr001" for repair_id, _support_keys in first_five)


def test_model_split_v2_prioritizes_bass_width_repair_before_pair_residual() -> None:
    pairs = stem_search._prioritized_model_split_support_repair_pairs(
        support_options=[
            {},
            {"bass": {"role": "bass_support"}},
        ],
        repair_options=[
            {"repair_id": "none", "type": "none"},
            {"repair_id": "pr16", "type": "delivery_pair_residual"},
            {"repair_id": "bs400", "type": "bass_stereo_width"},
        ],
    )

    ordered = [(repair["repair_id"], "bass" in support) for support, repair in pairs]

    assert ordered.index(("bs400", True)) < ordered.index(("pr16", True))


def test_model_split_v2_prioritizes_teacher_mask_lane_routes(tmp_path: Path) -> None:
    base_vocal = tmp_path / "base_vocal.flac"
    learned_vocal = tmp_path / "learned_vocal.flac"
    base_minus = tmp_path / "base_minus.flac"
    learned_minus = tmp_path / "learned_minus.flac"
    for path in (base_vocal, learned_vocal, base_minus, learned_minus):
        path.write_bytes(b"audio")

    window_candidates = [
        {
            "target_specific_composition": {
                "routes": [
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "audio_separator_bs_roformer_vocals",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(base_vocal)}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "audio_separator_bs_roformer_vocals:instrumental",
                        "copied_outputs": [{"stem": "minus", "source_path": str(base_minus)}],
                    },
                    {
                        "role": "vocals",
                        "lane": "teacher_mask_checkpoint_lane",
                        "route_id": "tm001_vocals",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(learned_vocal)}],
                    },
                    {
                        "role": "minus",
                        "lane": "teacher_mask_checkpoint_lane",
                        "route_id": "tm002_minus",
                        "copied_outputs": [{"stem": "minus", "source_path": str(learned_minus)}],
                    },
                ]
            }
        }
    ]

    vocal_routes = stem_search._model_split_routes_by_role(window_candidates, "vocals", ("vocals",))
    minus_routes = stem_search._model_split_routes_by_role(window_candidates, "minus", ("minus",))

    assert [route["route_id"] for route in vocal_routes] == [
        "tm001_vocals",
        "audio_separator_bs_roformer_vocals",
    ]
    assert [route["route_id"] for route in minus_routes] == [
        "tm002_minus",
        "audio_separator_bs_roformer_vocals:instrumental",
    ]


def test_model_split_v2_pair_ranking_tests_teacher_mask_pair_before_old_residual_pair(tmp_path: Path) -> None:
    frames = SAMPLE_RATE
    t = np.arange(frames, dtype=np.float32) / float(SAMPLE_RATE)
    old_vocal = 0.10 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    old_minus = 0.08 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    mix = old_vocal + old_minus
    learned_vocal = 0.04 * np.sin(2.0 * np.pi * 330.0 * t).astype(np.float32)
    learned_minus = 0.03 * np.sin(2.0 * np.pi * 550.0 * t).astype(np.float32)

    mix_path = write_stereo_wav(tmp_path / "mix.flac", mix, mix)
    old_vocal_path = write_stereo_wav(tmp_path / "old_vocal.flac", old_vocal, old_vocal)
    old_minus_path = write_stereo_wav(tmp_path / "old_minus.flac", old_minus, old_minus)
    learned_vocal_path = write_stereo_wav(tmp_path / "learned_vocal.flac", learned_vocal, learned_vocal)
    learned_minus_path = write_stereo_wav(tmp_path / "learned_minus.flac", learned_minus, learned_minus)

    old_vocal_route = {
        "role": "vocals",
        "lane": "vocal_lane",
        "route_id": "audio_separator_bs_roformer_vocals",
        "copied_outputs": [{"stem": "vocals", "source_path": str(old_vocal_path)}],
    }
    old_minus_route = {
        "role": "minus",
        "lane": "minus_lane",
        "route_id": "audio_separator_bs_roformer_vocals:instrumental",
        "copied_outputs": [{"stem": "minus", "source_path": str(old_minus_path)}],
    }
    learned_vocal_route = {
        "role": "vocals",
        "lane": "teacher_mask_checkpoint_lane",
        "route_id": "tm001_vocals",
        "copied_outputs": [{"stem": "vocals", "source_path": str(learned_vocal_path)}],
    }
    learned_minus_route = {
        "role": "minus",
        "lane": "teacher_mask_checkpoint_lane",
        "route_id": "tm002_minus",
        "copied_outputs": [{"stem": "minus", "source_path": str(learned_minus_path)}],
    }

    pairs = stem_search._rank_model_split_vocal_minus_pairs(
        mix_path=mix_path,
        vocal_routes=[learned_vocal_route, old_vocal_route],
        minus_routes=[learned_minus_route, old_minus_route],
    )

    first_vocal, first_minus, first_residual = pairs[0]

    assert first_vocal["route_id"] == "tm001_vocals"
    assert first_minus["route_id"] == "tm002_minus"
    assert first_residual > pairs[-1][2]


def test_parse_stem_region_repair_checkpoint_spec_preserves_windows_path() -> None:
    parsed = stem_search._parse_stem_region_repair_checkpoint_spec("bass:stereo_collapse:D:/models/model.pt")

    assert parsed == {
        "repair_id": "sr001",
        "stem": "bass",
        "issue_type": ISSUE_STEREO_COLLAPSE,
        "checkpoint": "D:/models/model.pt",
    }


def test_default_model_split_v2_composite_limit_respects_runtime_max_candidates() -> None:
    recipe = {"model_split_composite_max_candidates": 864}

    assert (
        stem_search._effective_model_split_composite_limit(
            recipe=recipe,
            max_candidates=10,
            default_recipe=True,
        )
        == 10
    )
    assert (
        stem_search._effective_model_split_composite_limit(
            recipe=recipe,
            max_candidates=10,
            default_recipe=False,
        )
        == 864
    )


def test_model_split_v2_htdemucs_ft_minus_uses_registered_cli_contract() -> None:
    recipe = build_model_split_v2_recipe(demucs_available=True)

    htdemucs_ft_minus = next(
        route
        for candidate in recipe["candidates"]
        for route in candidate["target_specific_commands"]
        if route["route_id"] == "htdemucs_ft:instrumental"
    )

    command_template = htdemucs_ft_minus["command_template"]
    assert command_template[:4] == [
        "python",
        "tools/run_registered_model_candidate.py",
        "{input_audio}",
        "{stems_dir}",
    ]
    assert "--input-audio" not in command_template
    assert "--output-dir" not in command_template
    assert command_template[command_template.index("--model-id") + 1] == "htdemucs_ft"
    assert command_template[command_template.index("--target-stem") + 1] == "minus"
    assert htdemucs_ft_minus["output_map"] == {"minus": "minus"}


def test_model_split_v2_bs_mamba2_uses_registered_vocal_minus_lanes(monkeypatch) -> None:
    monkeypatch.setattr(stem_search, "_registered_route_is_available", lambda route_id: True)
    recipe = build_model_split_v2_recipe(demucs_available=True)

    assert "bs_mamba2_vocals" in recipe["target_specific_routes"]["vocal_lane"]
    assert "bs_mamba2_vocals:instrumental" in recipe["target_specific_routes"]["minus_lane"]

    bs_mamba2_vocal = next(
        route
        for candidate in recipe["candidates"]
        for route in candidate["target_specific_commands"]
        if route["route_id"] == "bs_mamba2_vocals"
    )
    bs_mamba2_minus = next(
        route
        for candidate in recipe["candidates"]
        for route in candidate["target_specific_commands"]
        if route["route_id"] == "bs_mamba2_vocals:instrumental"
    )

    vocal_command = bs_mamba2_vocal["command_template"]
    minus_command = bs_mamba2_minus["command_template"]
    assert "tools/run_registered_model_candidate.py" in vocal_command
    assert "tools/run_registered_model_candidate.py" in minus_command
    assert vocal_command[vocal_command.index("--model-id") + 1] == "bs_mamba2_vocals"
    assert minus_command[minus_command.index("--model-id") + 1] == "bs_mamba2_vocals"
    assert vocal_command[vocal_command.index("--target-stem") + 1] == "vocals"
    assert minus_command[minus_command.index("--target-stem") + 1] == "minus"
    assert bs_mamba2_minus["output_map"] == {"minus": "instrumental"}


def test_model_split_v2_filters_unavailable_registered_routes(monkeypatch) -> None:
    def fake_available(route_id: str) -> bool:
        return not str(route_id).startswith("bs_mamba2_vocals")

    monkeypatch.setattr(stem_search, "_registered_route_is_available", fake_available)
    recipe = build_model_split_v2_recipe(demucs_available=True)

    assert "bs_mamba2_vocals" not in recipe["target_specific_routes"]["vocal_lane"]
    assert "bs_mamba2_vocals:instrumental" not in recipe["target_specific_routes"]["minus_lane"]
    assert all(
        route["route_id"] != "bs_mamba2_vocals"
        for candidate in recipe["candidates"]
        for route in candidate["target_specific_commands"]
    )
    assert all(
        route["route_id"] != "bs_mamba2_vocals:instrumental"
        for candidate in recipe["candidates"]
        for route in candidate["target_specific_commands"]
    )


def test_model_split_v2_minus_delivery_cannot_be_rebuilt_from_six_stems(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.06),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    rebuilt_minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    lane_minus = rebuilt_minus * np.float32(0.5)
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + lane_minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "delivery_pair_reconstruction": {"max_delivery_pair_residual_ratio": 2.0},
                        "target_specific_commands": [
                            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
                            {"role": "vocals", "lane": "vocal_lane", "command_template": ["fixture-vocal", "{window_mix}"]},
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "audio_separator_bs_roformer_vocals:instrumental",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        calls.append(list(command))
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-six":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus":
            sf.write(stems_dir / "instrumental.flac", lane_minus[: len(window_audio)], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    accepted_minus, _ = sf.read(candidate["stem_paths"]["minus"], dtype="float32", always_2d=False)
    assert result["verdict"] == "auto_accept"
    assert candidate["minus_source"] == "minus_lane"
    assert candidate["minus_rebuilt_from_non_vocal_stems"] is False
    np.testing.assert_allclose(accepted_minus, lane_minus, atol=2e-4)
    assert not np.allclose(accepted_minus, rebuilt_minus, atol=2e-4)
    assert result["listen_pack"]["lane_provenance"]["minus"]["lane"] == "minus_lane"
    assert candidate["reconstruction"]["mode"] == "model_split_v2_delivery_pair"
    assert candidate["reconstruction"]["delivery_pair"]["ok"] is True
    assert candidate["reconstruction"]["six_stem_diagnostic"]["ok"] is False
    minus_route = next(
        route
        for route in candidate["target_specific_composition"]["routes"]
        if route["role"] == "minus"
    )
    assert minus_route["route_id"] == "audio_separator_bs_roformer_vocals:instrumental"
    assert minus_route["route_alias"] == "audio_separator_bs_roformer_vocals_instrumental"
    assert ":" not in Path(minus_route["output_dir"]).name


def test_model_split_v2_blocks_delivery_pair_mismatch(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.06),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    rebuilt_minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    lane_minus = rebuilt_minus * np.float32(0.25)
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + rebuilt_minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "delivery_pair_fail_fast": True,
                        "delivery_pair_reconstruction": {"max_delivery_pair_residual_ratio": 0.06},
                        "target_specific_commands": [
                            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
                            {"role": "vocals", "lane": "vocal_lane", "command_template": ["fixture-vocal", "{window_mix}"]},
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        calls.append(list(command))
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-six":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus":
            sf.write(stems_dir / "instrumental.flac", lane_minus[: len(window_audio)], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["verdict"] == "blocked_bad_reconstruction"
    assert candidate["reconstruction"]["mode"] == "model_split_v2_delivery_pair"
    assert candidate["reconstruction"]["delivery_pair"]["ok"] is False
    assert candidate["delivery_pair_fail_fast"]["enabled"] is True
    assert "six_stems" in candidate["delivery_pair_fail_fast"]["skipped_roles"]
    assert [command[0] for command in calls] == ["fixture-vocal", "fixture-minus"]


def test_model_split_v2_mdx_route_uses_runtime_budget(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.06),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "timeout_sec": 300,
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "delivery_pair_fail_fast": True,
                        "delivery_pair_reconstruction": {"max_delivery_pair_residual_ratio": 0.06},
                        "target_specific_commands": [
                            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "mdx23c_vocals",
                                "command_template": ["fixture-vocal", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    timeouts_by_command: dict[str, int] = {}

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        timeouts_by_command[command[0]] = timeout_sec
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus":
            sf.write(stems_dir / "instrumental.flac", (minus * np.float32(0.25))[: len(window_audio)], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    mdx_route = next(route for route in candidate["target_specific_composition"]["routes"] if route["route_id"] == "mdx23c_vocals")
    assert timeouts_by_command["fixture-vocal"] == 45
    assert timeouts_by_command["fixture-minus"] == 300
    assert mdx_route["timeout_sec"] == 45
    assert mdx_route["separator_backend"]["route_timeout_sec"] == 45


def test_model_split_v2_fast_route_budget_caps_slow_delivery_pair_routes(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.06),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "timeout_sec": 300,
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "delivery_pair_fail_fast": True,
                        "delivery_pair_reconstruction": {"max_delivery_pair_residual_ratio": 0.06},
                        "target_specific_commands": [
                            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "mdx23c_vocals",
                                "command_template": ["fixture-vocal", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "becruily_instrumental_vocals",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    timeouts_by_command: dict[str, int] = {}

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        timeouts_by_command[command[0]] = timeout_sec
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus":
            sf.write(stems_dir / "instrumental.flac", (minus * np.float32(0.25))[: len(window_audio)], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
        route_budget_profile="fast",
    )

    routes = result["candidates"][0]["target_specific_composition"]["routes"]
    route_timeouts = {route["route_id"]: route["timeout_sec"] for route in routes}
    assert result["route_budget_profile"] == "fast"
    assert timeouts_by_command["fixture-vocal"] == 30
    assert timeouts_by_command["fixture-minus"] == 120
    assert route_timeouts["mdx23c_vocals"] == 30
    assert route_timeouts["becruily_instrumental_vocals"] == 120


def test_model_split_v2_mandatory_gates_cannot_be_disabled(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.06),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    base_candidate = {
        "profile": "model_split_v2",
        "minus_source": "minus_lane",
        "target_specific_commands": [
            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
            {"role": "vocals", "lane": "vocal_lane", "command_template": ["fixture-vocal", "{window_mix}"]},
            {
                "role": "minus",
                "lane": "minus_lane",
                "output_map": {"minus": "instrumental"},
                "command_template": ["fixture-minus", "{window_mix}"],
            },
        ],
        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
    }
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {**base_candidate, "candidate_id": "c001", "damage_gates": {"enabled": False}},
                    {**base_candidate, "candidate_id": "c002", "voice_presence_gate": {"enabled": False}},
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-six":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus":
            sf.write(stems_dir / "instrumental.flac", minus[: len(window_audio)], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=2,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    by_id = {candidate["candidate_id"]: candidate for candidate in result["candidates"]}
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert by_id["c001"]["verdict"] == "blocked_damage"
    assert "v2_damage_gate_disabled" in by_id["c001"]["damage_score"]["blocker_reasons"]
    assert by_id["c002"]["verdict"] == "blocked_unknown"
    assert "v2_voice_presence_gate_disabled" in by_id["c002"]["voice_presence_score"]["blocker_reasons"]


def test_model_split_v2_cross_stem_forensic_blocks_vocal_drum_imprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drums = cymbal_like_drums(seconds=0.5)
    source_audio = {
        "vocals": bright_voice_like_vocal(seconds=0.5) + 0.18 * drums,
        "drums": drums,
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "delivery_pair_reconstruction": {"max_delivery_pair_residual_ratio": 2.0},
                        "target_specific_commands": [
                            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
                            {"role": "vocals", "lane": "vocal_lane", "command_template": ["fixture-vocal", "{window_mix}"]},
                            {"role": "minus", "lane": "minus_lane", "command_template": ["fixture-minus", "{window_mix}"]},
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def old_detector_pass(**_kwargs: object) -> dict[str, object]:
        return {"verdict": "auto_accept", "attempted_checks": 1, "blocker_reasons": [], "leak_results": []}

    monkeypatch.setattr(stem_search, "_score_candidate_contamination", old_detector_pass)

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-six":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus":
            sf.write(stems_dir / "minus.flac", minus[: len(window_audio)], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["verdict"] == "blocked_leak"
    assert candidate["delivery_allowed"] is False
    assert "vocal_contains_drum_transient_imprint" in candidate["cross_stem_forensic_score"]["blocker_reasons"]
    assert candidate["cross_stem_forensic_score"]["verdict"] == "fail"
    assert candidate["contamination_score"]["verdict"] == "auto_accept"
    assert result["accepted_window_candidates"] == {}
    assert not (tmp_path / "scratch" / "t001" / "listen").exists()


def test_listen_pack_forensic_failure_deletes_listen_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_audio = {
        "vocals": bright_voice_like_vocal(seconds=0.5),
        "drums": cymbal_like_drums(seconds=0.5),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "delivery_pair_reconstruction": {"max_delivery_pair_residual_ratio": 2.0},
                        "target_specific_commands": [
                            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
                            {"role": "vocals", "lane": "vocal_lane", "command_template": ["fixture-vocal", "{window_mix}"]},
                            {"role": "minus", "lane": "minus_lane", "command_template": ["fixture-minus", "{window_mix}"]},
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        stem_search,
        "_score_candidate_contamination",
        lambda **_kwargs: {"verdict": "auto_accept", "attempted_checks": 1, "blocker_reasons": [], "leak_results": []},
    )
    monkeypatch.setattr(
        stem_search,
        "score_cross_stem_forensic_matrix",
        lambda **_kwargs: {"verdict": "auto_accept", "blocker_reasons": [], "pair_results": []},
    )
    monkeypatch.setattr(
        stem_search,
        "validate_listen_pack_forensic",
        lambda **_kwargs: {
            "verdict": "fail",
            "blocker_reasons": ["vocal_contains_drum_transient_imprint"],
            "pair_results": [],
        },
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-six":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus":
            sf.write(stems_dir / "minus.flac", minus[: len(window_audio)], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "blocked_unknown"
    assert result["delivery_allowed"] is False
    assert result["presentation_status"] == "blocked_from_listen_pack"
    assert result["failed_checks"] == ["listen_pack_forensic_failed"]
    assert result["listen_pack"] == {}
    assert result["listen_pack_forensic_score"]["blocker_reasons"] == ["vocal_contains_drum_transient_imprint"]
    assert not (tmp_path / "scratch" / "t001" / "listen").exists()


def test_listen_pack_forensic_uses_accepted_candidate_gate_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_audio = {
        "vocals": bright_voice_like_vocal(seconds=0.5),
        "drums": cymbal_like_drums(seconds=0.5),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    candidate_forensic_gate = {
        "probe_gains": [1.0, 3.0],
        "vocal_drum_score_threshold": 0.91,
    }
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "delivery_pair_reconstruction": {"max_delivery_pair_residual_ratio": 2.0},
                        "cross_stem_forensic_gate": candidate_forensic_gate,
                        "target_specific_commands": [
                            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
                            {"role": "vocals", "lane": "vocal_lane", "command_template": ["fixture-vocal", "{window_mix}"]},
                            {"role": "minus", "lane": "minus_lane", "command_template": ["fixture-minus", "{window_mix}"]},
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        stem_search,
        "_score_candidate_contamination",
        lambda **_kwargs: {"verdict": "auto_accept", "attempted_checks": 1, "blocker_reasons": [], "leak_results": []},
    )
    candidate_configs: list[dict] = []
    listen_configs: list[dict] = []

    def candidate_forensic_pass(**kwargs: object) -> dict[str, object]:
        candidate_configs.append(dict(kwargs.get("config") or {}))
        return {"verdict": "auto_accept", "blocker_reasons": [], "pair_results": []}

    def listen_forensic_pass(**kwargs: object) -> dict[str, object]:
        listen_configs.append(dict(kwargs.get("config") or {}))
        return {"verdict": "auto_accept", "blocker_reasons": [], "pair_results": []}

    monkeypatch.setattr(stem_search, "score_cross_stem_forensic_matrix", candidate_forensic_pass)
    monkeypatch.setattr(stem_search, "validate_listen_pack_forensic", listen_forensic_pass)

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-six":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus":
            sf.write(stems_dir / "minus.flac", minus[: len(window_audio)], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    assert result["delivery_allowed"] is True
    assert candidate_configs == [candidate_forensic_gate]
    assert listen_configs == [candidate_forensic_gate]


def test_model_split_v2_builds_composite_from_independent_lane_winners(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.06),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_good",
                                "command_template": ["fixture-six", "{window_mix}", "{output_dir}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_bad",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-bad", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    },
                    {
                        "candidate_id": "c002",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_good",
                                "command_template": ["fixture-six", "{window_mix}", "{output_dir}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_bad",
                                "command_template": ["fixture-vocal-bad", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_good",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-good", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-six":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)], mix_rate)
        elif command[0] == "fixture-vocal-bad":
            sf.write(stems_dir / "vocals.flac", (source_audio["vocals"] * np.float32(0.5))[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus-good":
            sf.write(stems_dir / "instrumental.flac", minus[: len(window_audio)], mix_rate)
        elif command[0] == "fixture-minus-bad":
            sf.write(stems_dir / "instrumental.flac", (minus * np.float32(0.25))[: len(window_audio)], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=2,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    assert result["accepted_candidate_id"].startswith("c10")
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert accepted["model_split_composite"] is True
    lanes = accepted["target_specific_composition"]["routes"]
    assert next(route for route in lanes if route["role"] == "vocals")["source_candidate_id"] == "c001"
    assert next(route for route in lanes if route["role"] == "minus")["source_candidate_id"] == "c002"
    reused_stem_route = next(
        route for route in result["candidates"][1]["target_specific_composition"]["routes"] if route["role"] == "six_stems"
    )
    assert reused_stem_route["reused_route"] is True
    assert reused_stem_route["reused_from_candidate_id"] == "c001"
    assert accepted["reconstruction"]["delivery_pair"]["ok"] is True
    assert result["listen_pack"]["lane_provenance"]["vocals"]["route_id"] == "vocal_good"
    assert result["listen_pack"]["lane_provenance"]["minus"]["route_id"] == "minus_good"


def test_model_split_v2_composite_can_use_independent_non_vocal_support_lane(tmp_path: Path) -> None:
    def wide(mono: np.ndarray) -> np.ndarray:
        return np.stack([mono, mono * np.float32(0.2)], axis=1).astype(np.float32)

    def mono_stereo(mono: np.ndarray) -> np.ndarray:
        return np.stack([mono, mono], axis=1).astype(np.float32)

    source_audio = {
        "vocals": mono_stereo(sine(880.0, seconds=0.5, amp=0.12)),
        "drums": wide(sine(110.0, seconds=0.5, amp=0.06)),
        "bass": wide(sine(80.0, seconds=0.5, amp=0.05)),
        "guitar": wide(sine(330.0, seconds=0.5, amp=0.04)),
        "piano": wide(sine(440.0, seconds=0.5, amp=0.03)),
        "other": wide(sine(550.0, seconds=0.5, amp=0.02)),
    }
    bad_bass = mono_stereo(sine(80.0, seconds=0.5, amp=0.05))
    bad_stems = dict(source_audio)
    bad_stems["bass"] = bad_bass
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "model_split_composite_max_candidates": 8,
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "damage_gates": {"enabled": True, "min_relative_stereo_width": 0.5},
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_bad_bass",
                                "command_template": ["fixture-six-bad-bass", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_bad",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-bad", "{window_mix}"],
                            },
                            {
                                "role": "non_vocal_support",
                                "lane": "non_vocal_support_lane",
                                "route_id": "scnet_xl_ihf_musdb18:drums_bass_other",
                                "command_template": ["fixture-non-vocal-support", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    },
                    {
                        "candidate_id": "c002",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "damage_gates": {"enabled": True, "min_relative_stereo_width": 0.5},
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_bad_bass",
                                "command_template": ["fixture-six-bad-bass", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_bad",
                                "command_template": ["fixture-vocal-bad", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_good",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-good", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=True)
        stems_dir = fake_runner.stems_dir
        frames = len(window_audio)
        if command[0] == "fixture-six-bad-bass":
            for stem, audio in bad_stems.items():
                sf.write(stems_dir / f"{stem}.flac", audio[:frames], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][:frames], mix_rate)
        elif command[0] == "fixture-vocal-bad":
            sf.write(stems_dir / "vocals.flac", np.zeros_like(source_audio["vocals"][:frames]), mix_rate)
        elif command[0] == "fixture-minus-good":
            sf.write(stems_dir / "instrumental.flac", minus[:frames], mix_rate)
        elif command[0] == "fixture-minus-bad":
            sf.write(stems_dir / "instrumental.flac", np.zeros_like(minus[:frames]), mix_rate)
        elif command[0] == "fixture-non-vocal-support":
            for stem in ("drums", "bass", "other"):
                sf.write(stems_dir / f"{stem}.flac", source_audio[stem][:frames], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=2,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert accepted["bass_support_applied"] is True
    assert accepted["stem_support_applied"] is True
    assert accepted["non_vocal_support_applied"] is True
    assert accepted["target_specific_composition"]["bass_support_applied"] is True
    assert accepted["target_specific_composition"]["non_vocal_support_applied"] is True
    routes = accepted["target_specific_composition"]["routes"]
    assert next(route for route in routes if route["role"] == "non_vocal_support")["lane"] == "non_vocal_support_lane"
    assert accepted["damage_score"]["verdict"] == "auto_accept"
    assert "stereo_width_collapse:bass" not in accepted["damage_score"]["blocker_reasons"]


def test_model_split_v2_composite_projection_repair_can_clean_vocal_leak(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.06),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    drum_leak = source_audio["vocals"] * np.float32(0.04)
    other_leak = source_audio["vocals"] * np.float32(0.03)
    vocal_lane = source_audio["vocals"]
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "model_split_composite_max_candidates": 2,
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "model_split_composite_repair_options": [
                            {"repair_id": "none", "type": "none"},
                            {
                                "repair_id": "vp100",
                                "type": "vocal_projection_debleed",
                                "strength": 1.0,
                                "target_stems": ["drums", "other"],
                            },
                        ],
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_dirty_vocal_bleed",
                                "command_template": ["fixture-six-dirty-vocal-bleed", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_good",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-good", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        frames = len(window_audio)
        if command[0] == "fixture-six-dirty-vocal-bleed":
            sf.write(stems_dir / "vocals.flac", vocal_lane[:frames], mix_rate)
            sf.write(stems_dir / "drums.flac", (source_audio["drums"] + drum_leak)[:frames], mix_rate)
            sf.write(stems_dir / "bass.flac", source_audio["bass"][:frames], mix_rate)
            sf.write(stems_dir / "guitar.flac", source_audio["guitar"][:frames], mix_rate)
            sf.write(stems_dir / "piano.flac", source_audio["piano"][:frames], mix_rate)
            sf.write(stems_dir / "other.flac", (source_audio["other"] + other_leak)[:frames], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", vocal_lane[:frames], mix_rate)
        elif command[0] == "fixture-minus-good":
            sf.write(stems_dir / "instrumental.flac", minus[:frames], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=1,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    rejected_none = next(
        candidate
        for candidate in result["candidates"]
        if candidate.get("model_split_composite") and candidate.get("model_split_repair_id") == "none"
    )
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert rejected_none["verdict"] == "blocked_leak"
    assert accepted["model_split_repair_id"] == "vp100"
    assert accepted["model_split_repair_type"] == "vocal_projection_debleed"
    assert accepted["vocal_projection_debleed"]["applied"] is True
    assert accepted["contamination_score"]["verdict"] == "auto_accept"
    assert accepted["reconstruction"]["delivery_pair"]["ok"] is True


def test_model_split_v2_composite_bass_stereo_repair_can_clear_damage_gate(tmp_path: Path) -> None:
    def stereo_pair(mono: np.ndarray, right_gain: float) -> np.ndarray:
        return np.stack([mono, mono * np.float32(right_gain)], axis=1).astype(np.float32)

    source_audio = {
        "vocals": stereo_pair(sine(880.0, seconds=0.5, amp=0.03), 0.6),
        "drums": stereo_pair(sine(110.0, seconds=0.5, amp=0.07), 0.4),
        "bass": stereo_pair(sine(80.0, seconds=0.5, amp=0.06), -0.7),
        "guitar": stereo_pair(sine(330.0, seconds=0.5, amp=0.04), 0.3),
        "piano": stereo_pair(sine(440.0, seconds=0.5, amp=0.03), 0.5),
        "other": stereo_pair(sine(550.0, seconds=0.5, amp=0.02), 0.7),
    }
    damaged_bass = stereo_pair(sine(80.0, seconds=0.5, amp=0.06), 0.8)
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "model_split_composite_max_candidates": 2,
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "damage_gates": {"enabled": True},
                        "model_split_composite_repair_options": [
                            {"repair_id": "none", "type": "none"},
                            {"repair_id": "bs400", "type": "bass_stereo_width", "side_gain": 4.0},
                        ],
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_bass_narrow",
                                "command_template": ["fixture-six-bass-narrow", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_good",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-good", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=True)
        stems_dir = fake_runner.stems_dir
        frames = len(window_audio)
        if command[0] == "fixture-six-bass-narrow":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", (damaged_bass if stem == "bass" else audio)[:frames], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][:frames], mix_rate)
        elif command[0] == "fixture-minus-good":
            sf.write(stems_dir / "instrumental.flac", minus[:frames], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=1,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    rejected_none = next(
        candidate
        for candidate in result["candidates"]
        if candidate.get("model_split_composite") and candidate.get("model_split_repair_id") == "none"
    )
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert rejected_none["verdict"] == "blocked_damage"
    assert "stereo_width_collapse:bass" in rejected_none["damage_score"]["blocker_reasons"]
    assert accepted["model_split_repair_id"] == "bs400"
    assert accepted["bass_stereo_width_repair"]["applied"] is True
    assert accepted["bass_stereo_width_repair"]["after_width"] > accepted["bass_stereo_width_repair"]["before_width"]
    assert accepted["damage_score"]["verdict"] == "auto_accept"


def test_model_split_v2_composite_learned_stem_region_repair_can_clear_damage_gate(tmp_path: Path) -> None:
    def stereo_pair(mono: np.ndarray, side: np.ndarray) -> np.ndarray:
        return np.stack([mono + side, mono - side], axis=1).astype(np.float32)

    bass_mid = sine(80.0, seconds=0.5, amp=0.06)
    bass_side = sine(113.0, seconds=0.5, amp=0.025)
    repaired_target_bass = stereo_pair(bass_mid, bass_side)
    damaged_bass = stereo_pair(bass_mid, bass_side * np.float32(0.02))
    source_audio = {
        "vocals": stereo_pair(sine(880.0, seconds=0.5, amp=0.03), sine(910.0, seconds=0.5, amp=0.004)),
        "drums": stereo_pair(sine(110.0, seconds=0.5, amp=0.07), sine(150.0, seconds=0.5, amp=0.01)),
        "bass": repaired_target_bass,
        "guitar": stereo_pair(sine(330.0, seconds=0.5, amp=0.04), sine(350.0, seconds=0.5, amp=0.008)),
        "piano": stereo_pair(sine(440.0, seconds=0.5, amp=0.03), sine(460.0, seconds=0.5, amp=0.008)),
        "other": stereo_pair(sine(550.0, seconds=0.5, amp=0.02), sine(570.0, seconds=0.5, amp=0.006)),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix_audio = source_audio["vocals"] + minus
    mix = write_wav(tmp_path / "mix.wav", mix_audio)
    mask = make_region_mask(
        frame_count=damaged_bass.shape[0],
        sample_rate=SAMPLE_RATE,
        start_sec=0.0,
        end_sec=damaged_bass.shape[0] / SAMPLE_RATE,
    )
    model, history = train_region_repair_model_from_arrays(
        mix_audio=mix_audio,
        bad_stem_audio=damaged_bass,
        target_stem_audio=repaired_target_bass,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type=ISSUE_STEREO_COLLAPSE,
        config=StemRegionRepairTrainConfig(epochs=80, hidden_channels=16, learning_rate=0.01, device="cpu"),
    )
    checkpoint = save_region_repair_checkpoint(model=model, path=tmp_path / "model" / "model.pt", training_history=history)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "model_split_composite_max_candidates": 2,
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "damage_gates": {"enabled": True},
                        "model_split_composite_repair_options": [
                            {"repair_id": "none", "type": "none"},
                            {
                                "repair_id": "sr001",
                                "type": "stem_region_repair",
                                "stem": "bass",
                                "issue_type": ISSUE_STEREO_COLLAPSE,
                                "checkpoint": str(checkpoint),
                            },
                        ],
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_bass_collapsed",
                                "command_template": ["fixture-six-bass-collapsed", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_good",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-good", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=True)
        stems_dir = fake_runner.stems_dir
        frames = len(window_audio)
        if command[0] == "fixture-six-bass-collapsed":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", (damaged_bass if stem == "bass" else audio)[:frames], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][:frames], mix_rate)
        elif command[0] == "fixture-minus-good":
            sf.write(stems_dir / "instrumental.flac", minus[:frames], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=1,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert accepted["model_split_repair_id"] == "sr001"
    assert accepted["stem_region_repair"]["applied"] is True
    assert accepted["stem_region_repair"]["diagnostics"]["target_issue_improved"] is True
    assert accepted["damage_score"]["verdict"] == "auto_accept"


def test_learned_stem_region_repair_uses_checkpoint_region_mask(tmp_path: Path) -> None:
    def stereo_pair(mono: np.ndarray, side: np.ndarray) -> np.ndarray:
        return np.stack([mono + side, mono - side], axis=1).astype(np.float32)

    clean = stereo_pair(sine(440.0, seconds=0.5, amp=0.04), sine(460.0, seconds=0.5, amp=0.01))
    detail = stereo_pair(sine(880.0, seconds=0.5, amp=0.015), sine(910.0, seconds=0.5, amp=0.004))
    sibling = stereo_pair(sine(110.0, seconds=0.5, amp=0.05), sine(130.0, seconds=0.5, amp=0.01))
    mask = make_region_mask(frame_count=clean.shape[0], sample_rate=SAMPLE_RATE, start_sec=0.10, end_sec=0.24)
    target = clean + detail * mask[:, None]
    bad = clean
    mix_audio = target + sibling
    model, history = train_region_repair_model_from_arrays(
        mix_audio=mix_audio,
        bad_stem_audio=bad,
        target_stem_audio=target,
        sibling_stem_audio=sibling,
        target_mix_audio=mix_audio,
        region_mask=mask,
        sample_rate=SAMPLE_RATE,
        issue_type=ISSUE_BAD_RECONSTRUCTION,
        config=StemRegionRepairTrainConfig(
            epochs=60,
            hidden_channels=16,
            learning_rate=0.01,
            mix_conservation_loss_weight=1.0,
            device="cpu",
        ),
    )
    history = {**history, "region": {"start_sec": 0.10, "end_sec": 0.24}}
    checkpoint = save_region_repair_checkpoint(model=model, path=tmp_path / "model" / "model.pt", training_history=history)
    mix_path = write_wav(tmp_path / "mix.flac", mix_audio)
    piano_path = write_wav(tmp_path / "piano.flac", bad)

    result = stem_search.apply_learned_stem_region_repair(
        mix_path=mix_path,
        stem_paths={"piano": str(piano_path)},
        config={"stem": "piano", "issue_type": ISSUE_BAD_RECONSTRUCTION, "checkpoint": str(checkpoint)},
    )

    expected_active = int(round((0.24 - 0.10) * SAMPLE_RATE))
    assert result["applied"] is True
    assert result["region"] == {"start_sec": 0.10, "end_sec": 0.24}
    assert result["diagnostics"]["active_frame_count"] == pytest.approx(expected_active, abs=1)


def test_bad_reconstruction_vocal_repair_can_improve_delivery_pair_residual(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = int(SAMPLE_RATE * 0.25)
    detail = np.stack(
        [
            sine(440.0, seconds=0.25, amp=0.02),
            sine(450.0, seconds=0.25, amp=0.02),
        ],
        axis=1,
    ).astype(np.float32)
    vocals_bad = np.zeros((frames, 2), dtype=np.float32)
    minus_audio = np.zeros((frames, 2), dtype=np.float32)
    mix_audio = detail.copy()
    mix_path = write_wav(tmp_path / "mix.flac", mix_audio)
    vocals_path = write_wav(tmp_path / "vocals.flac", vocals_bad)
    minus_path = write_wav(tmp_path / "minus.flac", minus_audio)
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    class DummyModel:
        input_channels = 6

    def fake_load_region_repair_checkpoint(_path: Path):
        return DummyModel(), {
            "training_history": {
                "issue_type": ISSUE_BAD_RECONSTRUCTION,
                "region": {"start_sec": 0.0, "end_sec": frames / SAMPLE_RATE},
            }
        }

    def fake_apply_region_repair_model(**_kwargs):
        return detail.copy(), {
            "schema": "fake_repair",
            "active_frame_count": frames,
            "changed_region_ratio": 0.02,
            "changed_outside_ratio": 0.0,
        }

    monkeypatch.setattr(stem_search, "load_region_repair_checkpoint", fake_load_region_repair_checkpoint)
    monkeypatch.setattr(stem_search, "apply_region_repair_model", fake_apply_region_repair_model)

    before_residual = float(np.sqrt(np.mean(np.square(mix_audio - (vocals_bad + minus_audio)))))
    result = stem_search.apply_learned_stem_region_repair(
        mix_path=mix_path,
        stem_paths={"vocals": str(vocals_path), "minus": str(minus_path)},
        config={
            "stem": "vocals",
            "issue_type": ISSUE_BAD_RECONSTRUCTION,
            "checkpoint": str(checkpoint_path),
        },
    )
    repaired_vocals, _ = sf.read(vocals_path, dtype="float32", always_2d=True)
    repaired_minus, _ = sf.read(minus_path, dtype="float32", always_2d=True)
    after_residual = float(np.sqrt(np.mean(np.square(mix_audio - (repaired_vocals + repaired_minus)))))

    assert result["applied"] is True
    assert result["delivery_pair_compensation_stem"] == ""
    assert result["delivery_pair_compensation_delta_rms"] == 0.0
    assert result["diagnostics"]["delivery_pair_residual_before"] == pytest.approx(before_residual, abs=1.0e-4)
    assert result["diagnostics"]["delivery_pair_residual_after"] == pytest.approx(after_residual, abs=1.0e-4)
    assert result["diagnostics"]["target_issue_improved"] is True
    assert after_residual < before_residual * 0.25


def test_model_split_v2_composite_pair_gain_repair_can_clear_reconstruction_gate(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    weak_minus = minus * np.float32(0.93)
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "model_split_composite_max_candidates": 2,
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "model_split_composite_repair_options": [
                            {"repair_id": "none", "type": "none"},
                            {
                                "repair_id": "pg08",
                                "type": "delivery_pair_gain",
                                "target_stem": "minus",
                                "max_abs_gain_delta": 0.08,
                            },
                        ],
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_good",
                                "command_template": ["fixture-six-good", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_weak",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-weak", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        frames = len(window_audio)
        if command[0] == "fixture-six-good":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[:frames], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][:frames], mix_rate)
        elif command[0] == "fixture-minus-weak":
            sf.write(stems_dir / "instrumental.flac", weak_minus[:frames], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=1,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    rejected_none = next(
        candidate
        for candidate in result["candidates"]
        if candidate.get("model_split_composite") and candidate.get("model_split_repair_id") == "none"
    )
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert rejected_none["verdict"] == "blocked_bad_reconstruction"
    assert accepted["model_split_repair_id"] == "pg08"
    assert accepted["delivery_pair_gain_repair"]["applied"] is True
    assert accepted["delivery_pair_gain_repair"]["after_residual_ratio"] < accepted["delivery_pair_gain_repair"]["before_residual_ratio"]
    assert accepted["reconstruction"]["delivery_pair"]["ok"] is True


def test_model_split_v2_composite_pair_residual_repair_can_clear_reconstruction_gate(tmp_path: Path) -> None:
    residual = sine(760.0, seconds=0.5, amp=0.008)
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02) + residual,
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    missing_residual_minus = minus - residual
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "model_split_composite_max_candidates": 2,
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "model_split_composite_repair_options": [
                            {"repair_id": "none", "type": "none"},
                            {
                                "repair_id": "pr08",
                                "type": "delivery_pair_residual",
                                "target_stem": "minus",
                                "max_residual_ratio": 0.08,
                            },
                        ],
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_good",
                                "command_template": ["fixture-six-good", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_missing_residual",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-missing-residual", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        frames = len(window_audio)
        if command[0] == "fixture-six-good":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[:frames], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][:frames], mix_rate)
        elif command[0] == "fixture-minus-missing-residual":
            sf.write(stems_dir / "instrumental.flac", missing_residual_minus[:frames], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=1,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    rejected_none = next(
        candidate
        for candidate in result["candidates"]
        if candidate.get("model_split_composite") and candidate.get("model_split_repair_id") == "none"
    )
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert rejected_none["verdict"] == "blocked_bad_reconstruction"
    assert accepted["model_split_repair_id"] == "pr08"
    assert accepted["delivery_pair_residual_repair"]["applied"] is True
    assert accepted["delivery_pair_residual_repair"]["after_residual_ratio"] < accepted["delivery_pair_residual_repair"]["before_residual_ratio"]
    assert accepted["reconstruction"]["delivery_pair"]["ok"] is True


def test_model_split_v2_composite_builder_reports_inner_progress(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    stem_dir = tmp_path / "routes" / "stems"
    vocal_dir = tmp_path / "routes" / "vocal"
    minus_dir = tmp_path / "routes" / "minus"
    for stem, audio in source_audio.items():
        write_wav(stem_dir / f"{stem}.flac", audio)
    write_wav(vocal_dir / "vocals.flac", source_audio["vocals"])
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    write_wav(minus_dir / "minus.flac", minus)
    window_mix = write_wav(tmp_path / "w000.wav", source_audio["vocals"] + minus)
    window_candidates = [
        {
            "target_specific_composition": {
                "routes": [
                    {
                        "role": "six_stems",
                        "lane": "stem_lane",
                        "route_id": "stem_fixture",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(stem_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "guitar", "piano", "other")
                        ],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "vocal_fixture",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(vocal_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "minus_fixture",
                        "copied_outputs": [{"stem": "minus", "source_path": str(minus_dir / "minus.flac")}],
                    },
                ]
            }
        }
    ]
    events: list[dict] = []

    result = stem_search.build_model_split_v2_composite_candidates(
        window={
            "window_id": "w000",
            "mix_path": str(window_mix),
            "sample_rate": SAMPLE_RATE,
            "sample_count": len(source_audio["vocals"]),
            "channel_count": 1,
        },
        window_dir=tmp_path / "run" / "w000",
        window_candidates=window_candidates,
        candidate_template={
            "profile": "model_split_v2",
            "foreign_references": write_full_foreign_references(tmp_path, source_audio),
        },
        max_composites=1,
        progress_callback=events.append,
    )

    assert result
    assert events[0]["event"] == "composite_candidate_start"
    assert events[0]["window_id"] == "w000"
    assert events[0]["candidate_id"] == "c101"
    assert events[-1]["event"] == "composite_candidate_end"
    assert events[-1]["candidate_id"] == "c101"
    assert events[-1]["verdict"] in VERDICTS
    starts = [event for event in events if event["event"] == "composite_candidate_start"]
    ends = [event for event in events if event["event"] == "composite_candidate_end"]
    assert [event["candidate_id"] for event in starts] == ["c101"]
    assert [event["candidate_id"] for event in ends] == ["c101"]
    assert len({(event["event"], event["candidate_id"]) for event in starts + ends}) == 2

    events.clear()
    limited = stem_search.build_model_split_v2_composite_candidates(
        window={"window_id": "w000", "mix_path": str(window_mix)},
        window_dir=tmp_path / "run_limit" / "w000",
        window_candidates=window_candidates,
        candidate_template={
            "profile": "model_split_v2",
            "foreign_references": write_full_foreign_references(tmp_path, source_audio),
        },
        max_composites=0,
        progress_callback=events.append,
    )

    assert limited == []
    assert events == [{"event": "composite_candidate_limit_reached", "window_id": "w000", "max_composites": 0}]


def test_model_split_v2_composite_reaches_support_routes_before_repair_exhaustion(tmp_path: Path) -> None:
    def wide(mono: np.ndarray) -> np.ndarray:
        return np.stack([mono, mono * np.float32(0.2)], axis=1).astype(np.float32)

    def mono_stereo(mono: np.ndarray) -> np.ndarray:
        return np.stack([mono, mono], axis=1).astype(np.float32)

    source_audio = {
        "vocals": mono_stereo(sine(880.0, seconds=0.5, amp=0.03)),
        "drums": wide(sine(110.0, seconds=0.5, amp=0.07)),
        "bass": wide(sine(220.0, seconds=0.5, amp=0.05)),
        "guitar": wide(sine(330.0, seconds=0.5, amp=0.04)),
        "piano": wide(sine(440.0, seconds=0.5, amp=0.03)),
        "other": wide(sine(550.0, seconds=0.5, amp=0.02)),
    }
    damaged_stem_audio = dict(source_audio)
    damaged_stem_audio["bass"] = mono_stereo(sine(220.0, seconds=0.5, amp=0.05))
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    window_mix = write_wav(tmp_path / "w000.wav", source_audio["vocals"] + minus)
    stem_dir = tmp_path / "routes" / "stems"
    vocal_dir = tmp_path / "routes" / "vocal"
    minus_dir = tmp_path / "routes" / "minus"
    bass_support_dir = tmp_path / "routes" / "bass_support"
    other_support_dir = tmp_path / "routes" / "other_support"
    non_vocal_support_dir = tmp_path / "routes" / "non_vocal_support"
    for stem, audio in damaged_stem_audio.items():
        write_wav(stem_dir / f"{stem}.flac", audio)
    write_wav(vocal_dir / "vocals.flac", source_audio["vocals"])
    write_wav(minus_dir / "minus.flac", minus)
    write_wav(bass_support_dir / "bass.flac", source_audio["bass"])
    write_wav(other_support_dir / "other.flac", source_audio["other"])
    for stem in ("drums", "bass", "other"):
        write_wav(non_vocal_support_dir / f"{stem}.flac", source_audio[stem])
    window_candidates = [
        {
            "target_specific_composition": {
                "routes": [
                    {
                        "role": "six_stems",
                        "lane": "stem_lane",
                        "route_id": "stem_fixture",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(stem_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "guitar", "piano", "other")
                        ],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "vocal_fixture",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(vocal_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "minus_fixture",
                        "copied_outputs": [{"stem": "minus", "source_path": str(minus_dir / "minus.flac")}],
                    },
                    {
                        "role": "bass_support",
                        "lane": "bass_support_lane",
                        "route_id": "bass_support_fixture",
                        "copied_outputs": [{"stem": "bass", "source_path": str(bass_support_dir / "bass.flac")}],
                    },
                    {
                        "role": "other_support",
                        "lane": "teacher_mask_checkpoint_lane",
                        "route_id": "other_support_fixture",
                        "copied_outputs": [{"stem": "other", "source_path": str(other_support_dir / "other.flac")}],
                    },
                    {
                        "role": "non_vocal_support",
                        "lane": "non_vocal_support_lane",
                        "route_id": "non_vocal_support_fixture",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(non_vocal_support_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "other")
                        ],
                    },
                ]
            }
        }
    ]
    repair_options = [
        {"repair_id": "none", "type": "none"},
        {"repair_id": "pg01", "type": "delivery_pair_gain", "max_gain": 1.01},
        {"repair_id": "pg02", "type": "delivery_pair_gain", "max_gain": 1.02},
        {"repair_id": "pg03", "type": "delivery_pair_gain", "max_gain": 1.03},
        {"repair_id": "pg04", "type": "delivery_pair_gain", "max_gain": 1.04},
    ]
    events: list[dict] = []

    result = stem_search.build_model_split_v2_composite_candidates(
        window={
            "window_id": "w000",
            "mix_path": str(window_mix),
            "sample_rate": SAMPLE_RATE,
            "sample_count": len(source_audio["vocals"]),
            "channel_count": 1,
        },
        window_dir=tmp_path / "run" / "w000",
        window_candidates=window_candidates,
        candidate_template={
            "profile": "model_split_v2",
            "damage_gates": {"enabled": True, "min_relative_stereo_width": 0.5},
            "foreign_references": write_full_foreign_references(tmp_path, source_audio),
            "model_split_composite_repair_options": repair_options,
        },
        max_composites=4,
        progress_callback=events.append,
    )

    assert [candidate["model_split_repair_id"] for candidate in result] == ["none", "none", "none", "none"]
    assert [candidate["stem_support_applied"] for candidate in result] == [False, True, True, True]
    assert result[1]["bass_support_applied"] is True
    assert result[2]["target_specific_composition"]["routes"][-1]["role"] == "other_support"
    assert result[3]["non_vocal_support_applied"] is True
    starts = [event for event in events if event["event"] == "composite_candidate_start"]
    assert [event["repair_id"] for event in starts] == ["none", "none", "none", "none"]


def test_model_split_v2_composite_can_combine_learned_single_stem_support_routes(tmp_path: Path) -> None:
    def wide(mono: np.ndarray) -> np.ndarray:
        return np.stack([mono, mono * np.float32(0.2)], axis=1).astype(np.float32)

    source_audio = {
        "vocals": wide(sine(880.0, seconds=0.5, amp=0.03)),
        "drums": wide(sine(110.0, seconds=0.5, amp=0.07)),
        "bass": wide(sine(220.0, seconds=0.5, amp=0.05)),
        "guitar": wide(sine(330.0, seconds=0.5, amp=0.04)),
        "piano": wide(sine(440.0, seconds=0.5, amp=0.03)),
        "other": wide(sine(550.0, seconds=0.5, amp=0.02)),
    }
    damaged_stem_audio = dict(source_audio)
    damaged_stem_audio["guitar"] = np.zeros_like(source_audio["guitar"])
    damaged_stem_audio["other"] = np.zeros_like(source_audio["other"])
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    window_mix = write_wav(tmp_path / "w000.wav", source_audio["vocals"] + minus)
    stem_dir = tmp_path / "routes" / "stems"
    vocal_dir = tmp_path / "routes" / "vocal"
    minus_dir = tmp_path / "routes" / "minus"
    guitar_support_dir = tmp_path / "routes" / "guitar_support"
    other_support_dir = tmp_path / "routes" / "other_support"
    for stem, audio in damaged_stem_audio.items():
        write_wav(stem_dir / f"{stem}.flac", audio)
    write_wav(vocal_dir / "vocals.flac", source_audio["vocals"])
    write_wav(minus_dir / "minus.flac", minus)
    write_wav(guitar_support_dir / "guitar.flac", source_audio["guitar"])
    write_wav(other_support_dir / "other.flac", source_audio["other"])
    window_candidates = [
        {
            "target_specific_composition": {
                "routes": [
                    {
                        "role": "six_stems",
                        "lane": "stem_lane",
                        "route_id": "stem_fixture",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(stem_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "guitar", "piano", "other")
                        ],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "vocal_fixture",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(vocal_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "minus_fixture",
                        "copied_outputs": [{"stem": "minus", "source_path": str(minus_dir / "minus.flac")}],
                    },
                    {
                        "role": "guitar_support",
                        "lane": "teacher_mask_checkpoint_lane",
                        "route_id": "tm_guitar_support",
                        "copied_outputs": [{"stem": "guitar", "source_path": str(guitar_support_dir / "guitar.flac")}],
                    },
                    {
                        "role": "other_support",
                        "lane": "teacher_mask_checkpoint_lane",
                        "route_id": "tm_other_support",
                        "copied_outputs": [{"stem": "other", "source_path": str(other_support_dir / "other.flac")}],
                    },
                ]
            }
        }
    ]

    result = stem_search.build_model_split_v2_composite_candidates(
        window={
            "window_id": "w000",
            "mix_path": str(window_mix),
            "sample_rate": SAMPLE_RATE,
            "sample_count": len(source_audio["vocals"]),
            "channel_count": 2,
        },
        window_dir=tmp_path / "run" / "w000",
        window_candidates=window_candidates,
        candidate_template={
            "profile": "model_split_v2",
            "foreign_references": write_full_foreign_references(tmp_path, source_audio),
            "model_split_composite_repair_options": [{"repair_id": "none", "type": "none"}],
        },
        max_composites=5,
    )

    combined = next(
        (
            candidate
            for candidate in result
            if {"guitar_support", "other_support"}.issubset(
                {route["role"] for route in candidate["target_specific_composition"]["routes"]}
            )
        ),
        None,
    )

    assert combined is not None
    support_routes = {
        route["role"]: route
        for route in combined["target_specific_composition"]["routes"]
        if route["role"] in {"guitar_support", "other_support"}
    }
    assert support_routes["guitar_support"]["lane"] == "teacher_mask_checkpoint_lane"
    assert support_routes["other_support"]["lane"] == "teacher_mask_checkpoint_lane"
    assert support_routes["guitar_support"]["copied_outputs"] == [
        {
            "stem": "guitar",
            "source_stem": "guitar",
            "source_path": str(guitar_support_dir / "guitar.flac"),
            "output_path": str(Path(combined["stems_dir"]) / "guitar.flac"),
        }
    ]
    assert support_routes["other_support"]["copied_outputs"] == [
        {
            "stem": "other",
            "source_stem": "other",
            "source_path": str(other_support_dir / "other.flac"),
            "output_path": str(Path(combined["stems_dir"]) / "other.flac"),
        }
    ]


def test_model_split_v2_composite_reaches_support_key_repairs_before_budget_exhaustion(tmp_path: Path) -> None:
    def wide(mono: np.ndarray) -> np.ndarray:
        return np.stack([mono, mono * np.float32(0.2)], axis=1).astype(np.float32)

    def mono_stereo(mono: np.ndarray) -> np.ndarray:
        return np.stack([mono, mono], axis=1).astype(np.float32)

    source_audio = {
        "vocals": mono_stereo(sine(880.0, seconds=0.5, amp=0.03)),
        "drums": wide(sine(110.0, seconds=0.5, amp=0.07)),
        "bass": wide(sine(220.0, seconds=0.5, amp=0.05)),
        "guitar": wide(sine(330.0, seconds=0.5, amp=0.04)),
        "piano": wide(sine(440.0, seconds=0.5, amp=0.03)),
        "other": wide(sine(550.0, seconds=0.5, amp=0.02)),
    }
    damaged_stem_audio = dict(source_audio)
    damaged_stem_audio["bass"] = mono_stereo(sine(220.0, seconds=0.5, amp=0.05))
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    window_mix = write_wav(tmp_path / "w000.wav", source_audio["vocals"] + minus)
    stem_dir = tmp_path / "routes" / "stems"
    vocal_dir = tmp_path / "routes" / "vocal"
    minus_dir = tmp_path / "routes" / "minus"
    bass_support_dir = tmp_path / "routes" / "bass_support"
    non_vocal_support_dir = tmp_path / "routes" / "non_vocal_support"
    for stem, audio in damaged_stem_audio.items():
        write_wav(stem_dir / f"{stem}.flac", audio)
    write_wav(vocal_dir / "vocals.flac", source_audio["vocals"])
    write_wav(minus_dir / "minus.flac", minus)
    write_wav(bass_support_dir / "bass.flac", source_audio["bass"])
    for stem in ("drums", "bass", "other"):
        write_wav(non_vocal_support_dir / f"{stem}.flac", source_audio[stem])
    window_candidates = [
        {
            "target_specific_composition": {
                "routes": [
                    {
                        "role": "six_stems",
                        "lane": "stem_lane",
                        "route_id": "stem_fixture",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(stem_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "guitar", "piano", "other")
                        ],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "vocal_fixture",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(vocal_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "minus_fixture",
                        "copied_outputs": [{"stem": "minus", "source_path": str(minus_dir / "minus.flac")}],
                    },
                    {
                        "role": "bass_support",
                        "lane": "bass_support_lane",
                        "route_id": "bass_support_fixture",
                        "copied_outputs": [{"stem": "bass", "source_path": str(bass_support_dir / "bass.flac")}],
                    },
                    {
                        "role": "non_vocal_support",
                        "lane": "non_vocal_support_lane",
                        "route_id": "non_vocal_support_fixture",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(non_vocal_support_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "other")
                        ],
                    },
                ]
            }
        }
    ]
    repair_options = [
        {"repair_id": "none", "type": "none"},
        {"repair_id": "vp100", "type": "vocal_projection_debleed", "strength": 1.0},
        {"repair_id": "bs250", "type": "bass_stereo_width", "side_gain": 2.5},
        {"repair_id": "bs400", "type": "bass_stereo_width", "side_gain": 4.0},
        {"repair_id": "pg08", "type": "delivery_pair_gain", "target_stem": "minus", "max_abs_gain_delta": 0.08},
        {"repair_id": "pr16", "type": "delivery_pair_residual", "target_stem": "minus", "max_residual_ratio": 0.16},
        {
            "repair_id": "bs400_pr16",
            "type": "bass_stereo_width+delivery_pair_residual",
            "bass_stereo_width": {"side_gain": 4.0},
            "delivery_pair_residual": {"target_stem": "minus", "max_residual_ratio": 0.16},
        },
    ]
    events: list[dict] = []

    stem_search.build_model_split_v2_composite_candidates(
        window={
            "window_id": "w000",
            "mix_path": str(window_mix),
            "sample_rate": SAMPLE_RATE,
            "sample_count": len(source_audio["vocals"]),
            "channel_count": 2,
        },
        window_dir=tmp_path / "run" / "w000",
        window_candidates=window_candidates,
        candidate_template={
            "profile": "model_split_v2",
            "damage_gates": {"enabled": True, "min_relative_stereo_width": 10.0},
            "foreign_references": write_full_foreign_references(tmp_path, source_audio),
            "model_split_composite_repair_options": repair_options,
        },
        max_composites=10,
        progress_callback=events.append,
    )

    starts = [event for event in events if event["event"] == "composite_candidate_start"]
    first_ten = [
        (event["repair_id"], event["bass_support_applied"], event["non_vocal_support_applied"])
        for event in starts[:10]
    ]
    assert ("pr16", True, False) in first_ten
    assert ("pr16", True, True) in first_ten
    assert ("bs400_pr16", True, False) in first_ten
    assert ("vp100", True, False) in first_ten
    assert not any(repair_id in {"bs250", "pg08"} and not bass for repair_id, bass, _ in first_ten)


def test_model_split_v2_composite_prioritizes_low_residual_vocal_minus_pair(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = source_audio["vocals"] + minus
    window_mix = write_wav(tmp_path / "w000.wav", mix)
    stem_dir = tmp_path / "routes" / "stems"
    vocal_bad_dir = tmp_path / "routes" / "vocal_bad"
    vocal_good_dir = tmp_path / "routes" / "vocal_good"
    minus_bad_dir = tmp_path / "routes" / "minus_bad"
    minus_good_dir = tmp_path / "routes" / "minus_good"
    for stem, audio in source_audio.items():
        write_wav(stem_dir / f"{stem}.flac", audio)
    write_wav(vocal_bad_dir / "vocals.flac", np.zeros_like(source_audio["vocals"]))
    write_wav(vocal_good_dir / "vocals.flac", source_audio["vocals"])
    write_wav(minus_bad_dir / "minus.flac", np.zeros_like(minus))
    write_wav(minus_good_dir / "minus.flac", minus)
    window_candidates = [
        {
            "target_specific_composition": {
                "routes": [
                    {
                        "role": "six_stems",
                        "lane": "stem_lane",
                        "route_id": "stem_good",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(stem_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "guitar", "piano", "other")
                        ],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "vocal_bad_first",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(vocal_bad_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "minus_bad_first",
                        "copied_outputs": [{"stem": "minus", "source_path": str(minus_bad_dir / "minus.flac")}],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "vocal_good_later",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(vocal_good_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "minus_good_later",
                        "copied_outputs": [{"stem": "minus", "source_path": str(minus_good_dir / "minus.flac")}],
                    },
                ]
            }
        }
    ]
    events: list[dict] = []

    result = stem_search.build_model_split_v2_composite_candidates(
        window={"window_id": "w000", "mix_path": str(window_mix)},
        window_dir=tmp_path / "run" / "w000",
        window_candidates=window_candidates,
        candidate_template={
            "profile": "model_split_v2",
            "foreign_references": write_full_foreign_references(tmp_path, source_audio),
        },
        max_composites=1,
        progress_callback=events.append,
    )

    assert len(result) == 1
    candidate = result[0]
    routes = candidate["target_specific_composition"]["routes"]
    assert candidate["verdict"] == "auto_accept"
    assert next(route for route in routes if route["role"] == "vocals")["route_id"] == "vocal_good_later"
    assert next(route for route in routes if route["role"] == "minus")["route_id"] == "minus_good_later"
    assert candidate["model_split_pair_residual_ratio"] == pytest.approx(0.0, abs=1e-3)
    assert events[0]["pair_residual_ratio"] == pytest.approx(0.0, abs=1e-3)


def test_model_split_v2_composite_keeps_distinct_audio_for_same_route_id(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = source_audio["vocals"] + minus
    window_mix = write_wav(tmp_path / "w000.wav", mix)
    stem_dir = tmp_path / "routes" / "stems"
    vocal_bad_dir = tmp_path / "routes" / "vocal_bad"
    vocal_good_dir = tmp_path / "routes" / "vocal_good"
    minus_bad_dir = tmp_path / "routes" / "minus_bad"
    minus_good_dir = tmp_path / "routes" / "minus_good"
    for stem, audio in source_audio.items():
        write_wav(stem_dir / f"{stem}.flac", audio)
    write_wav(vocal_bad_dir / "vocals.flac", np.zeros_like(source_audio["vocals"]))
    write_wav(vocal_good_dir / "vocals.flac", source_audio["vocals"])
    write_wav(minus_bad_dir / "minus.flac", np.zeros_like(minus))
    write_wav(minus_good_dir / "minus.flac", minus)
    window_candidates = [
        {
            "target_specific_composition": {
                "routes": [
                    {
                        "role": "six_stems",
                        "lane": "stem_lane",
                        "route_id": "stem_good",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(stem_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "guitar", "piano", "other")
                        ],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "shared_vocal_model",
                        "source_candidate_id": "bad_first",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(vocal_bad_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "shared_minus_model",
                        "source_candidate_id": "bad_first",
                        "copied_outputs": [{"stem": "minus", "source_path": str(minus_bad_dir / "minus.flac")}],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "shared_vocal_model",
                        "source_candidate_id": "good_later",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(vocal_good_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "shared_minus_model",
                        "source_candidate_id": "good_later",
                        "copied_outputs": [{"stem": "minus", "source_path": str(minus_good_dir / "minus.flac")}],
                    },
                ]
            }
        }
    ]

    result = stem_search.build_model_split_v2_composite_candidates(
        window={"window_id": "w000", "mix_path": str(window_mix)},
        window_dir=tmp_path / "run" / "w000",
        window_candidates=window_candidates,
        candidate_template={
            "profile": "model_split_v2",
            "foreign_references": write_full_foreign_references(tmp_path, source_audio),
        },
        max_composites=1,
    )

    assert len(result) == 1
    candidate = result[0]
    routes = candidate["target_specific_composition"]["routes"]
    assert candidate["verdict"] == "auto_accept"
    assert next(route for route in routes if route["role"] == "vocals")["source_candidate_id"] == "good_later"
    assert next(route for route in routes if route["role"] == "minus")["source_candidate_id"] == "good_later"
    assert candidate["model_split_pair_residual_ratio"] == pytest.approx(0.0, abs=1e-3)


def test_model_split_v2_composite_pair_ranking_uses_aligned_output_paths(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = source_audio["vocals"] + minus
    window_mix = write_wav(tmp_path / "w000.wav", mix)
    stem_dir = tmp_path / "routes" / "stems"
    bad_dir = tmp_path / "routes" / "bad"
    good_raw_dir = tmp_path / "routes" / "good_raw"
    good_out_dir = tmp_path / "routes" / "good_out"
    for stem, audio in source_audio.items():
        write_wav(stem_dir / f"{stem}.flac", audio)
    write_wav(bad_dir / "vocals.flac", np.zeros_like(source_audio["vocals"]))
    write_wav(bad_dir / "minus.flac", np.zeros_like(minus))
    write_wav(good_raw_dir / "vocals.flac", np.concatenate([source_audio["vocals"], np.zeros(1000, dtype=np.float32)]))
    write_wav(good_raw_dir / "minus.flac", np.concatenate([minus, np.zeros(1000, dtype=np.float32)]))
    write_wav(good_out_dir / "vocals.flac", source_audio["vocals"])
    write_wav(good_out_dir / "minus.flac", minus)
    window_candidates = [
        {
            "target_specific_composition": {
                "routes": [
                    {
                        "role": "six_stems",
                        "lane": "stem_lane",
                        "route_id": "stem_good",
                        "copied_outputs": [
                            {"stem": stem, "source_path": str(stem_dir / f"{stem}.flac")}
                            for stem in ("drums", "bass", "guitar", "piano", "other")
                        ],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "bad_vocal_first",
                        "copied_outputs": [{"stem": "vocals", "source_path": str(bad_dir / "vocals.flac")}],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "bad_minus_first",
                        "copied_outputs": [{"stem": "minus", "source_path": str(bad_dir / "minus.flac")}],
                    },
                    {
                        "role": "vocals",
                        "lane": "vocal_lane",
                        "route_id": "good_vocal_later",
                        "source_candidate_id": "good_aligned",
                        "copied_outputs": [
                            {
                                "stem": "vocals",
                                "source_path": str(good_raw_dir / "vocals.flac"),
                                "output_path": str(good_out_dir / "vocals.flac"),
                            }
                        ],
                    },
                    {
                        "role": "minus",
                        "lane": "minus_lane",
                        "route_id": "good_minus_later",
                        "source_candidate_id": "good_aligned",
                        "copied_outputs": [
                            {
                                "stem": "minus",
                                "source_path": str(good_raw_dir / "minus.flac"),
                                "output_path": str(good_out_dir / "minus.flac"),
                            }
                        ],
                    },
                ]
            }
        }
    ]

    result = stem_search.build_model_split_v2_composite_candidates(
        window={
            "window_id": "w000",
            "mix_path": str(window_mix),
            "sample_rate": SAMPLE_RATE,
            "sample_count": len(mix),
            "channel_count": 1,
        },
        window_dir=tmp_path / "run" / "w000",
        window_candidates=window_candidates,
        candidate_template={
            "profile": "model_split_v2",
            "foreign_references": write_full_foreign_references(tmp_path, source_audio),
        },
        max_composites=1,
    )

    assert len(result) == 1
    candidate = result[0]
    routes = candidate["target_specific_composition"]["routes"]
    assert candidate["verdict"] == "auto_accept"
    assert next(route for route in routes if route["role"] == "vocals")["source_candidate_id"] == "good_aligned"
    assert next(route for route in routes if route["role"] == "minus")["source_candidate_id"] == "good_aligned"
    assert candidate["model_split_pair_residual_ratio"] == pytest.approx(0.0, abs=1e-3)


def test_model_split_v2_recipe_level_no_reference_manifest_can_accept_full_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.01),
        "drums": sine(110.0, seconds=0.5, amp=0.07),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    minus = sum(source_audio[stem] for stem in ("drums", "bass", "guitar", "piano", "other"))
    mix = write_wav(tmp_path / "mix.wav", source_audio["vocals"] + minus)
    model_manifest = tmp_path / "model" / "manifest.json"
    model_manifest.parent.mkdir(parents=True, exist_ok=True)
    model_manifest.write_text("{}", encoding="utf-8")
    calls: list[dict] = []

    def fake_score_stem_contamination(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        assert kwargs["foreign_reference_path"] is None
        assert kwargs["model_manifest_path"] == model_manifest
        return {
            "tested_stem": kwargs["tested_stem"],
            "foreign_source": kwargs["foreign_source"],
            "verdict": "pass",
            "leak_audio": "",
            "regions": [],
            "leak_dbfs": -90.0,
            "detector_mode": "fixture_no_reference_v1",
        }

    monkeypatch.setattr(stem_search, "score_stem_contamination", fake_score_stem_contamination)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "contamination_model_manifest": str(model_manifest),
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "lane": "stem_lane",
                                "route_id": "stem_good",
                                "command_template": ["fixture-six-good", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "lane": "vocal_lane",
                                "route_id": "vocal_good",
                                "command_template": ["fixture-vocal-good", "{window_mix}"],
                            },
                            {
                                "role": "minus",
                                "lane": "minus_lane",
                                "route_id": "minus_good",
                                "output_map": {"minus": "instrumental"},
                                "command_template": ["fixture-minus-good", "{window_mix}"],
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        frames = len(window_audio)
        if command[0] == "fixture-six-good":
            for stem, audio in source_audio.items():
                sf.write(stems_dir / f"{stem}.flac", audio[:frames], mix_rate)
        elif command[0] == "fixture-vocal-good":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][:frames], mix_rate)
        elif command[0] == "fixture-minus-good":
            sf.write(stems_dir / "instrumental.flac", minus[:frames], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=1,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "auto_accept"
    assert result["delivery_status"] == "accepted"
    assert result["delivery_allowed"] is True
    assert result["presentation_status"] == "accepted_for_listen_pack"
    assert len(calls) == sum(len(sources) for sources in FORBIDDEN_SOURCES.values())
    accepted = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == result["accepted_candidate_id"])
    assert accepted["contamination_score"]["check_mode"] == "full_forbidden_matrix"
    assert accepted["contamination_score"]["candidate_check_hints_used"] is True


def test_model_split_v2_blocks_t040_style_post_debleed_and_other_merge(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.06),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    mix = write_wav(tmp_path / "mix.wav", sum(source_audio.values()))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c010",
                        "profile": "model_split_v2",
                        "command_template": ["fixture-write-legacy-repair", "{window_mix}"],
                        "stem_merges": [{"source_stem": "other", "target_stem": "guitar", "mode": "move"}],
                        "post_minus_vocal_debleed": {
                            "strength": 1.0,
                            "low_hz": 700.0,
                            "high_hz": 1200.0,
                            "target_stems": ["drums"],
                            "passes": 1,
                        },
                        "damage_gates": {"enabled": True},
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        for stem, audio in source_audio.items():
            sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)], mix_rate)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["verdict"] == "blocked_damage"
    assert "v2_repair_route_not_publishable:stem_merges" in candidate["damage_score"]["blocker_reasons"]
    assert "v2_repair_route_not_publishable:post_minus_vocal_debleed" in candidate["damage_score"]["blocker_reasons"]
    assert result["listen_pack"] == {}


def test_model_split_v2_stereo_damage_blocks_even_when_contamination_passes(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.5, amp=0.02)
    left_drums = sine(110.0, seconds=0.5, amp=0.07)
    right_drums = -left_drums
    bass = sine(220.0, seconds=0.5, amp=0.05)
    guitar = sine(330.0, seconds=0.5, amp=0.04)
    piano = sine(440.0, seconds=0.5, amp=0.03)
    other = sine(550.0, seconds=0.5, amp=0.02)
    mono_non_drums = bass + guitar + piano + other
    mix = write_stereo_wav(tmp_path / "mix.wav", vocal + left_drums + mono_non_drums, vocal + right_drums + mono_non_drums)
    refs = write_full_foreign_references(
        tmp_path,
        {
            "vocals": vocal,
            "drums": left_drums,
            "bass": bass,
            "guitar": guitar,
            "piano": piano,
            "other": other,
        },
    )
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "minus_source": "minus_lane",
                        "delivery_pair_reconstruction": {"max_delivery_pair_residual_ratio": 2.0},
                        "target_specific_commands": [
                            {"role": "six_stems", "lane": "stem_lane", "command_template": ["fixture-six", "{window_mix}"]},
                            {"role": "vocals", "lane": "vocal_lane", "command_template": ["fixture-vocal", "{window_mix}"]},
                            {"role": "minus", "lane": "minus_lane", "command_template": ["fixture-minus", "{window_mix}"]},
                        ],
                        "damage_gates": {
                            "enabled": True,
                            "min_relative_stereo_width": 0.5,
                            "min_non_vocal_rms": 0.0,
                        },
                        "foreign_references": refs,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=True)
        stems_dir = fake_runner.stems_dir
        frames = len(window_audio)
        if command[0] == "fixture-six":
            sf.write(stems_dir / "vocals.flac", np.stack([vocal, vocal], axis=1)[:frames], mix_rate)
            sf.write(stems_dir / "drums.flac", np.stack([left_drums, right_drums], axis=1)[:frames], mix_rate)
            for stem, audio in (("bass", bass), ("guitar", guitar), ("piano", piano), ("other", other)):
                sf.write(stems_dir / f"{stem}.flac", np.stack([audio, audio], axis=1)[:frames], mix_rate)
        elif command[0] == "fixture-vocal":
            sf.write(stems_dir / "vocals.flac", np.stack([vocal, vocal], axis=1)[:frames], mix_rate)
        elif command[0] == "fixture-minus":
            collapsed = left_drums + mono_non_drums
            sf.write(stems_dir / "minus.flac", np.stack([collapsed, collapsed], axis=1)[:frames], mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["verdict"] == "blocked_damage"
    assert "stereo_width_collapse:minus" in candidate["damage_score"]["blocker_reasons"]


def test_model_split_v2_bass_stereo_uses_bass_band_reference(tmp_path: Path) -> None:
    bass = sine(80.0, seconds=0.5, amp=0.06)
    wide_high = sine(2200.0, seconds=0.5, amp=0.04)
    mix_path = write_stereo_wav(tmp_path / "mix.wav", bass + wide_high, bass - wide_high)
    bass_path = write_stereo_wav(tmp_path / "bass.wav", bass, bass)
    minus_path = write_stereo_wav(tmp_path / "minus.wav", bass + wide_high, bass - wide_high)

    score = stem_search.score_model_split_v2_damage_gates(
        candidate={
            "profile": "model_split_v2",
            "minus_source": "minus_lane",
            "damage_gates": {
                "enabled": True,
                "min_relative_stereo_width": 0.5,
                "min_non_vocal_rms": 0.0,
                "min_mix_stereo_width": 0.02,
            },
        },
        mix_path=mix_path,
        stem_paths={"bass": str(bass_path), "minus": str(minus_path)},
    )

    assert score["verdict"] == "auto_accept"
    assert "stereo_width_collapse:bass" not in score["blocker_reasons"]
    assert score["metrics"]["bass"]["stereo_reference_surface"] == "bass_band_mix"
    assert score["metrics"]["mix"]["stereo_width"] > 0.02
    assert score["metrics"]["mix"]["bass_band_stereo_width"] < 0.02


def test_model_split_v2_bass_stereo_blocks_when_bass_band_collapses(tmp_path: Path) -> None:
    bass = sine(80.0, seconds=0.5, amp=0.06)
    mix_path = write_stereo_wav(tmp_path / "mix.wav", bass, bass * 0.2)
    bass_path = write_stereo_wav(tmp_path / "bass.wav", bass * 0.6, bass * 0.6)
    minus_path = write_stereo_wav(tmp_path / "minus.wav", bass, bass * 0.2)

    score = stem_search.score_model_split_v2_damage_gates(
        candidate={
            "profile": "model_split_v2",
            "minus_source": "minus_lane",
            "damage_gates": {
                "enabled": True,
                "min_relative_stereo_width": 0.5,
                "min_non_vocal_rms": 0.0,
                "min_mix_stereo_width": 0.02,
            },
        },
        mix_path=mix_path,
        stem_paths={"bass": str(bass_path), "minus": str(minus_path)},
    )

    assert score["verdict"] == "blocked_damage"
    assert "stereo_width_collapse:bass" in score["blocker_reasons"]
    assert score["metrics"]["bass"]["stereo_reference_surface"] == "bass_band_mix"


def test_model_split_v2_other_cannot_be_silently_emptied(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.03),
        "drums": sine(110.0, seconds=0.5, amp=0.06),
        "bass": sine(220.0, seconds=0.5, amp=0.05),
        "guitar": sine(330.0, seconds=0.5, amp=0.04),
        "piano": sine(440.0, seconds=0.5, amp=0.03),
        "other": sine(550.0, seconds=0.5, amp=0.02),
    }
    mix = write_wav(tmp_path / "mix.wav", sum(source_audio.values()))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v2",
                "profile": "model_split_v2",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "profile": "model_split_v2",
                        "command_template": ["fixture-write-empty-other", "{window_mix}"],
                        "damage_gates": {"enabled": True, "min_non_vocal_rms": 1e-4},
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        for stem, audio in source_audio.items():
            output = np.zeros_like(audio, dtype=np.float32) if stem == "other" else audio
            sf.write(stems_dir / f"{stem}.flac", output[: len(window_audio)], mix_rate)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["verdict"] == "blocked_damage"
    assert "stem_detail_floor:other" in candidate["damage_score"]["blocker_reasons"]


def test_four_stem_non_vocal_role_overwrites_only_drums_bass_other(tmp_path: Path) -> None:
    role_dir = tmp_path / "role" / "stems"
    final_dir = tmp_path / "final"
    original = {
        "vocals": sine(880.0, amp=0.02),
        "drums": sine(100.0, amp=0.03),
        "bass": sine(140.0, amp=0.04),
        "guitar": sine(330.0, amp=0.05),
        "piano": sine(440.0, amp=0.06),
        "other": sine(550.0, amp=0.07),
    }
    replacement = {
        "drums": sine(111.0, amp=0.11),
        "bass": sine(155.0, amp=0.12),
        "other": sine(600.0, amp=0.13),
    }
    for stem, audio in original.items():
        write_wav(final_dir / f"{stem}.flac", audio)
    for stem, audio in replacement.items():
        write_wav(role_dir / f"{stem}.flac", audio)

    blockers = _copy_target_specific_outputs(
        role="four_stem_non_vocal",
        role_stems_dir=role_dir,
        final_stems_dir=final_dir,
        target_sample_rate=SAMPLE_RATE,
        target_sample_count=len(next(iter(original.values()))),
        target_channel_count=1,
    )

    assert blockers == []
    for stem, audio in replacement.items():
        actual, _ = sf.read(final_dir / f"{stem}.flac", dtype="float32", always_2d=False)
        np.testing.assert_allclose(actual, audio, atol=2e-4)
    for stem in ("vocals", "guitar", "piano"):
        actual, _ = sf.read(final_dir / f"{stem}.flac", dtype="float32", always_2d=False)
        np.testing.assert_allclose(actual, original[stem], atol=2e-4)


def test_contamination_model_manifest_applies_to_full_matrix() -> None:
    checks, hints_used = _candidate_contamination_checks(
        {
            "contamination_model_manifest": "artifacts/scd/m001/manifest.json",
            "contamination_checks": [
                {
                    "tested_stem": "minus",
                    "foreign_source": "vocals",
                    "model_manifest": "artifacts/scd/m002/manifest.json",
                }
            ],
        }
    )

    required_check_count = sum(len(sources) for sources in FORBIDDEN_SOURCES.values())
    assert hints_used is True
    assert len(checks) == required_check_count
    assert all(check["model_manifest"] for check in checks)
    assert next(
        check for check in checks if check["tested_stem"] == "minus" and check["foreign_source"] == "vocals"
    )["model_manifest"] == "artifacts/scd/m002/manifest.json"
    assert next(
        check for check in checks if check["tested_stem"] == "vocals" and check["foreign_source"] == "drums"
    )["model_manifest"] == "artifacts/scd/m001/manifest.json"


def test_explicit_only_contamination_checks_can_skip_default_model_manifest() -> None:
    checks, hints_used = _candidate_contamination_checks(
        {
            "contamination_model_manifest": "artifacts/scd/m001/manifest.json",
            "use_default_contamination_model_manifest": False,
            "contamination_check_mode": "explicit_only",
            "contamination_checks": [{"tested_stem": "minus", "foreign_source": "vocals"}],
        }
    )

    assert hints_used is True
    assert checks == [{"tested_stem": "minus", "foreign_source": "vocals"}]


def test_persistent_mps_pair_recipe_scores_only_minus_vocal_without_neural_manifest() -> None:
    recipe = stem_search.build_persistent_mps_minus_pair_recipe(
        contamination_model_manifest="artifacts/scd/m001/manifest.json",
    )

    candidate = recipe["candidates"][0]
    checks, hints_used = _candidate_contamination_checks(
        {
            **candidate,
            "contamination_model_manifest": recipe["contamination_model_manifest"],
        }
    )

    assert hints_used is True
    assert candidate["contamination_check_mode"] == "explicit_only"
    assert candidate["use_default_contamination_model_manifest"] is False
    assert checks == [{"tested_stem": "minus", "foreign_source": "vocals"}]


def test_no_default_separator_blocks_without_old_recipe(tmp_path: Path) -> None:
    mix = write_wav(tmp_path / "mix.wav", sine(220.0, seconds=0.25))
    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        demucs_available=False,
    )

    assert result["verdict"] == "blocked_no_available_separator"
    assert result["failed_checks"] == ["blocked_no_available_separator"]
    assert result["candidate_count"] == 0


def test_recipe_candidate_executes_and_blocks_without_contamination_proof(tmp_path: Path) -> None:
    mix = write_wav(tmp_path / "mix.wav", sine(220.0, seconds=0.25))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-six-stems"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        assert command == ["fixture-write-six-stems"]
        stems_dir = fake_runner.stems_dir
        mix_audio, mix_rate = sf.read(mix, dtype="float32", always_2d=False)
        for stem in STEMS:
            audio = mix_audio if stem == "vocals" else np.zeros_like(mix_audio, dtype=np.float32)
            sf.write(stems_dir / f"{stem}.flac", np.asarray(audio, dtype=np.float32), mix_rate)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["candidate_count"] == 1
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["candidate_id"] == "c001"
    assert candidate["execution_status"] == "completed"
    assert candidate["verdict"] == "blocked_unknown"
    assert Path(candidate["stem_paths"]["minus"]).exists()
    assert result["windows"][0]["window_id"] == "w000"


def test_recipe_candidate_blocks_clean_minus_check_when_full_matrix_unproven(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.25, amp=0.04)
    drums = sine(110.0, seconds=0.25, amp=0.12)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-clean", "{window_mix}"],
                        "contamination_checks": [{"tested_stem": "minus", "foreign_source": "vocals"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        assert command[0] == "fixture-write-clean"
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)])
        write_wav(stems_dir / "drums.flac", drums[: len(window_audio)])
        for stem in ("bass", "guitar", "piano", "other"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert result["delivery_allowed"] is False
    assert result["listen_pack"] == {}
    assert candidate["verdict"] == "blocked_unknown"
    assert candidate["contamination_score"]["check_mode"] == "full_forbidden_matrix"
    assert "contamination_detector_unknown:vocals:drums" in candidate["contamination_score"]["blocker_reasons"]
    assert find_leak_result(candidate, "minus", "vocals")["detector_mode"] == "no_reference_spectral_v1"


def test_recipe_level_contamination_model_manifest_reaches_candidate_checks(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.25, amp=0.04)
    drums = sine(110.0, seconds=0.25, amp=0.12)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums)
    recipe = tmp_path / "recipe.json"
    missing_manifest = tmp_path / "missing_model" / "manifest.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "contamination_model_manifest": str(missing_manifest),
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-clean-with-model", "{window_mix}"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)])
        write_wav(stems_dir / "drums.flac", drums[: len(window_audio)])
        for stem in ("bass", "guitar", "piano", "other"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["contamination_score"]["candidate_check_hints_used"] is True
    assert candidate["contamination_score"]["attempted_checks"] == sum(len(sources) for sources in FORBIDDEN_SOURCES.values())
    assert any("no_reference_model_unavailable" in reason for reason in candidate["contamination_score"]["blocker_reasons"])


def test_recipe_candidate_accepts_only_with_full_forbidden_matrix_references(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(100.0, seconds=0.5, amp=0.03),
        "drums": sine(200.0, seconds=0.5, amp=0.04),
        "bass": sine(300.0, seconds=0.5, amp=0.05),
        "guitar": sine(400.0, seconds=0.5, amp=0.02),
        "piano": sine(500.0, seconds=0.5, amp=0.02),
        "other": sine(600.0, seconds=0.5, amp=0.01),
    }
    mix = write_wav(tmp_path / "mix.wav", sum(source_audio.values()))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-full-matrix-clean", "{window_mix}"],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        for stem, audio in source_audio.items():
            sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)].astype(np.float32), mix_rate)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    required_check_count = sum(len(sources) for sources in FORBIDDEN_SOURCES.values())
    assert result["verdict"] == "auto_accept"
    assert result["delivery_allowed"] is True
    assert candidate["verdict"] == "auto_accept"
    assert candidate["contamination_score"]["check_mode"] == "full_forbidden_matrix"
    assert candidate["contamination_score"]["attempted_checks"] == required_check_count
    assert candidate["contamination_score"]["blocker_reasons"] == []


def test_target_specific_vocal_route_overrides_six_stem_vocals(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.04),
        "drums": sine(110.0, seconds=0.5, amp=0.12),
        "bass": sine(220.0, seconds=0.5, amp=0.08),
        "guitar": sine(330.0, seconds=0.5, amp=0.06),
        "piano": sine(440.0, seconds=0.5, amp=0.05),
        "other": sine(550.0, seconds=0.5, amp=0.03),
    }
    bad_vocal = sine(660.0, seconds=0.5, amp=0.03)
    mix = write_wav(tmp_path / "mix.wav", sum(source_audio.values()))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "command_template": ["fixture-write-six-stem-route", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "command_template": ["fixture-write-vocal-route", "{window_mix}"],
                            },
                        ],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-write-six-stem-route":
            for stem, audio in source_audio.items():
                route_audio = bad_vocal if stem == "vocals" else audio
                sf.write(stems_dir / f"{stem}.flac", route_audio[: len(window_audio)].astype(np.float32), mix_rate)
        elif command[0] == "fixture-write-vocal-route":
            sf.write(stems_dir / "vocals.flac", source_audio["vocals"][: len(window_audio)].astype(np.float32), mix_rate)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    accepted_vocal, _ = sf.read(candidate["stem_paths"]["vocals"], dtype="float32", always_2d=False)
    assert result["verdict"] == "auto_accept"
    assert candidate["target_specific_composition"]["applied"] is True
    assert [route["role"] for route in candidate["target_specific_composition"]["routes"]] == ["six_stems", "vocals"]
    np.testing.assert_allclose(accepted_vocal, source_audio["vocals"], atol=2e-4)


def test_target_specific_missing_route_output_blocks(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.25, amp=0.04)
    drums = sine(110.0, seconds=0.25, amp=0.12)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "target_specific_commands": [
                            {
                                "role": "six_stems",
                                "command_template": ["fixture-write-six-stem-route", "{window_mix}"],
                            },
                            {
                                "role": "vocals",
                                "command_template": ["fixture-write-missing-vocal-route", "{window_mix}"],
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        if command[0] == "fixture-write-six-stem-route":
            write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)])
            write_wav(stems_dir / "drums.flac", drums[: len(window_audio)])
            for stem in ("bass", "guitar", "piano", "other"):
                sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)
        elif command[0] == "fixture-write-missing-vocal-route":
            stems_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise AssertionError(command)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["verdict"] == "blocked_missing_stem"
    assert "missing_target_specific_output:vocals:vocals" in candidate["target_specific_composition"]["blocker_reasons"]


def test_target_specific_output_is_normalized_to_window_shape(tmp_path: Path) -> None:
    source_dir = tmp_path / "route" / "stems"
    final_dir = tmp_path / "final"
    low_rate = SAMPLE_RATE // 2
    low_t = np.arange(int(low_rate * 0.5), dtype=np.float32) / float(low_rate)
    low_rate_vocal = (0.04 * np.sin(2.0 * math.pi * 440.0 * low_t)).astype(np.float32)
    source_dir.mkdir(parents=True, exist_ok=True)
    sf.write(source_dir / "vocals.flac", low_rate_vocal, low_rate)

    blockers = _copy_target_specific_outputs(
        role="vocals",
        role_stems_dir=source_dir,
        final_stems_dir=final_dir,
        target_sample_rate=SAMPLE_RATE,
        target_sample_count=int(SAMPLE_RATE * 0.5),
    )

    info = sf.info(final_dir / "vocals.flac")
    assert blockers == []
    assert int(info.samplerate) == SAMPLE_RATE
    assert int(info.frames) == int(SAMPLE_RATE * 0.5)


def test_listen_pack_build_failure_removes_partial_listen_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_audio = {
        "vocals": sine(100.0, seconds=0.5, amp=0.03),
        "drums": sine(200.0, seconds=0.5, amp=0.04),
        "bass": sine(300.0, seconds=0.5, amp=0.05),
        "guitar": sine(400.0, seconds=0.5, amp=0.02),
        "piano": sine(500.0, seconds=0.5, amp=0.02),
        "other": sine(600.0, seconds=0.5, amp=0.01),
    }
    mix = write_wav(tmp_path / "mix.wav", sum(source_audio.values()))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-accepted", "{window_mix}"],
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        for stem, audio in source_audio.items():
            sf.write(stems_dir / f"{stem}.flac", audio[: len(window_audio)].astype(np.float32), mix_rate)

    original_write_audio = stem_search._write_audio

    def fail_on_listen_drums(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
        if path.name == "drums.flac" and "listen" in path.parts:
            raise RuntimeError("forced listen pack write failure")
        return original_write_audio(path, audio, sample_rate)

    monkeypatch.setattr(stem_search, "_write_audio", fail_on_listen_drums)
    out = tmp_path / "scratch" / "t001"

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=out,
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    assert result["verdict"] == "blocked_unknown"
    assert result["delivery_status"] == "blocked"
    assert result["presentation_status"] == "blocked_from_listen_pack"
    assert result["listen_pack"] == {}
    assert result["listen_pack_error"]["message"] == "forced listen pack write failure"
    assert not (out / "listen").exists()


def test_recipe_candidate_blocks_no_reference_vocal_leak_in_minus(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.25, amp=0.04)
    drums = sine(110.0, seconds=0.25, amp=0.12)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-leaky", "{window_mix}"],
                        "contamination_checks": [{"tested_stem": "minus", "foreign_source": "vocals"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        assert command[0] == "fixture-write-leaky"
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        leak = 0.5 * vocal[: len(window_audio)]
        stems_dir = fake_runner.stems_dir
        write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)] - leak)
        write_wav(stems_dir / "drums.flac", drums[: len(window_audio)] + leak)
        for stem in ("bass", "guitar", "piano", "other"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    leak_result = find_leak_result(candidate, "minus", "vocals")
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["verdict"] == "blocked_leak"
    assert leak_result["verdict"] == "fail"
    assert leak_result["leak_audio"]
    assert Path(candidate["output_dir"], "score", leak_result["leak_audio"]).exists()


def test_recipe_candidate_uses_trainable_no_reference_model(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    rows = []
    for idx, freq in enumerate((820.0, 880.0, 960.0), start=1):
        clean = sine(110.0 + idx * 20.0, seconds=0.25, amp=0.18) + sine(220.0, seconds=0.25, amp=0.04)
        leak = sine(freq, seconds=0.25, amp=0.045)
        row_id = f"d{idx:06d}"
        rows.append(
            {
                "row_id": row_id,
                "input_stem": str(write_wav(tmp_path / "train" / f"{row_id}_in.flac", clean + leak)),
                "target_leak": str(write_wav(tmp_path / "train" / f"{row_id}_leak.flac", leak)),
                "base_stem": "minus",
                "foreign_source": "vocals",
                "label_type": "synthetic_clean",
                "is_ground_truth": True,
                "split": "train",
            }
        )
    rows_jsonl = tmp_path / "rows.jsonl"
    rows_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    train_stem_contamination_detector(
        rows_jsonl=rows_jsonl,
        output_dir=model_dir,
        device="cpu",
        epochs=120,
        hidden_size=8,
        min_positive_rows=2,
    )

    vocal = sine(880.0, seconds=0.25, amp=0.04)
    drums = sine(110.0, seconds=0.25, amp=0.12)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-model-clean", "{window_mix}"],
                        "contamination_checks": [
                            {
                                "tested_stem": "minus",
                                "foreign_source": "vocals",
                                "model_manifest": str(model_dir / "manifest.json"),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)])
        write_wav(stems_dir / "drums.flac", drums[: len(window_audio)])
        for stem in ("bass", "guitar", "piano", "other"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    leak_result = find_leak_result(candidate, "minus", "vocals")
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["verdict"] == "blocked_unknown"
    assert leak_result["detector_mode"] == "no_reference_neural_v1"
    assert leak_result["model_manifest"] == str(model_dir / "manifest.json")
    assert "contamination_detector_unknown:vocals:drums" in candidate["contamination_score"]["blocker_reasons"]


def test_conservation_repair_assigns_residual_before_reconstruction_gate(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.25, amp=0.04)
    drums = sine(110.0, seconds=0.25, amp=0.12)
    missing_other = sine(330.0, seconds=0.25, amp=0.03)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums + missing_other)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-needs-repair", "{window_mix}"],
                        "conservation_repair_stem": "other",
                        "foreign_references": write_full_foreign_references(
                            tmp_path,
                            {
                                "vocals": vocal,
                                "drums": drums,
                                "bass": np.zeros_like(vocal, dtype=np.float32),
                                "guitar": np.zeros_like(vocal, dtype=np.float32),
                                "piano": np.zeros_like(vocal, dtype=np.float32),
                                "other": missing_other,
                            },
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)])
        write_wav(stems_dir / "drums.flac", drums[: len(window_audio)])
        for stem in ("bass", "guitar", "piano", "other"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["verdict"] == "auto_accept"
    assert candidate["reconstruction_repair"]["applied"] is True
    assert candidate["reconstruction_repair"]["repair_owner"] == "runner_conservation_repair_v1"
    assert candidate["reconstruction_repair"]["target_stem"] == "other"
    assert candidate["reconstruction_repair"]["pre_repair_residual_rms"] > 0.0
    assert candidate["reconstruction"]["ok"] is True


def test_conservation_repair_rejects_vocal_target(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.25, amp=0.04)
    drums = sine(110.0, seconds=0.25, amp=0.12)
    missing_other = sine(330.0, seconds=0.25, amp=0.03)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums + missing_other)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-unsafe-repair", "{window_mix}"],
                        "conservation_repair_stem": "vocals",
                        "contamination_checks": [{"tested_stem": "minus", "foreign_source": "vocals"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)])
        write_wav(stems_dir / "drums.flac", drums[: len(window_audio)])
        for stem in ("bass", "guitar", "piano", "other"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["verdict"] == "blocked_bad_reconstruction"
    assert candidate["reconstruction_repair"]["applied"] is False
    assert candidate["reconstruction_repair"]["reason"] == "unsupported_conservation_repair_stem"


def test_conservation_repair_rejects_excessive_residual_ratio(tmp_path: Path) -> None:
    vocal = sine(880.0, seconds=0.25, amp=0.02)
    drums = sine(110.0, seconds=0.25, amp=0.02)
    missing_other = sine(330.0, seconds=0.25, amp=0.15)
    mix = write_wav(tmp_path / "mix.wav", vocal + drums + missing_other)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-excessive-repair", "{window_mix}"],
                        "conservation_repair_stem": "other",
                        "max_conservation_repair_ratio": 0.10,
                        "foreign_references": write_full_foreign_references(
                            tmp_path,
                            {
                                "vocals": vocal,
                                "drums": drums,
                                "bass": np.zeros_like(vocal, dtype=np.float32),
                                "guitar": np.zeros_like(vocal, dtype=np.float32),
                                "piano": np.zeros_like(vocal, dtype=np.float32),
                                "other": missing_other,
                            },
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        write_wav(stems_dir / "vocals.flac", vocal[: len(window_audio)])
        write_wav(stems_dir / "drums.flac", drums[: len(window_audio)])
        for stem in ("bass", "guitar", "piano", "other"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert candidate["verdict"] == "blocked_bad_reconstruction"
    assert candidate["reconstruction_repair"]["applied"] is False
    assert candidate["reconstruction_repair"]["reason"] == "conservation_repair_ratio_too_high"
    assert candidate["reconstruction_repair"]["max_residual_ratio"] == 0.10


def test_minus_vocal_debleed_does_not_publish_without_full_matrix_proof(tmp_path: Path) -> None:
    vocal_band_residue = sine(900.0, seconds=0.25, amp=0.04)
    bass = sine(110.0, seconds=0.25, amp=0.12)
    mix = write_wav(tmp_path / "mix.wav", vocal_band_residue + bass)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-debleedable", "{window_mix}"],
                        "minus_vocal_debleed": {"strength": 1.0, "low_hz": 700.0, "high_hz": 1200.0},
                        "contamination_checks": [{"tested_stem": "minus", "foreign_source": "vocals"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        sf.write(stems_dir / "vocals.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)
        write_wav(stems_dir / "bass.flac", bass[: len(window_audio)])
        write_wav(stems_dir / "other.flac", vocal_band_residue[: len(window_audio)])
        for stem in ("drums", "guitar", "piano"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["verdict"] == "blocked_no_clean_candidate"
    assert result["delivery_status"] == "blocked"
    assert result["delivery_allowed"] is False
    assert result["presentation_status"] == "blocked_from_listen_pack"
    assert result["listen_pack"] == {}
    assert candidate["minus_vocal_debleed"]["applied"] is True
    assert candidate["minus_vocal_debleed"]["moved_rms"] > 0.0
    assert candidate["reconstruction"]["ok"] is True
    assert candidate["contamination_score"]["verdict"] == "blocked_unknown"


def test_minus_vocal_debleed_passes_iteratively_reduce_vocal_band(tmp_path: Path) -> None:
    vocals = sine(880.0, seconds=0.5, amp=0.03)
    drums = sine(110.0, seconds=0.5, amp=0.08)
    bass = sine(55.0, seconds=0.5, amp=0.04)
    vocal_residue = sine(900.0, seconds=0.5, amp=0.05)
    paths = {
        "vocals": str(write_wav(tmp_path / "vocals.flac", vocals)),
        "drums": str(write_wav(tmp_path / "drums.flac", drums + vocal_residue)),
        "bass": str(write_wav(tmp_path / "bass.flac", bass)),
    }
    for stem in ("guitar", "piano", "other"):
        paths[stem] = str(write_wav(tmp_path / f"{stem}.flac", np.zeros_like(vocals, dtype=np.float32)))

    result = apply_minus_vocal_debleed(
        stem_paths=paths,
        config={
            "strength": 1.0,
            "low_hz": 700.0,
            "high_hz": 1200.0,
            "target_stems": ["drums"],
            "passes": 2,
        },
    )

    cleaned_drums, _ = sf.read(paths["drums"], dtype="float32", always_2d=False)
    before_band_rms = np.sqrt(np.mean(np.square(vocal_residue.astype(np.float64))))
    after_band_input = cleaned_drums[:, None] if cleaned_drums.ndim == 1 else cleaned_drums
    after_band = band_limited_audio(after_band_input, SAMPLE_RATE, 700.0, 1200.0)
    after_band_rms = np.sqrt(np.mean(np.square(after_band.astype(np.float64))))
    assert result["applied"] is True
    assert result["passes"] == 2
    assert result["per_stem"]["drums"]["passes"] == 2
    assert after_band_rms < before_band_rms * 0.05


def test_minus_vocal_debleed_invalid_passes_blocks_fail_closed(tmp_path: Path) -> None:
    audio = sine(880.0, seconds=0.25, amp=0.03)
    paths = {
        "vocals": str(write_wav(tmp_path / "vocals.flac", audio)),
        "drums": str(write_wav(tmp_path / "drums.flac", audio)),
    }

    result = apply_minus_vocal_debleed(
        stem_paths=paths,
        config={
            "strength": 1.0,
            "low_hz": 700.0,
            "high_hz": 1200.0,
            "target_stems": ["drums"],
            "passes": 0,
        },
    )

    assert result["applied"] is False
    assert result["reason"] == "invalid_debleed_config"
    assert "invalid_debleed_passes" in result["blockers"]


def test_recipe_candidate_applies_post_debleed_before_rebuilt_minus_scoring(tmp_path: Path) -> None:
    vocal_residue = sine(900.0, seconds=0.25, amp=0.05)
    drums = sine(110.0, seconds=0.25, amp=0.08)
    source_audio = {
        "vocals": vocal_residue,
        "drums": drums,
        "bass": np.zeros_like(vocal_residue, dtype=np.float32),
        "guitar": np.zeros_like(vocal_residue, dtype=np.float32),
        "piano": np.zeros_like(vocal_residue, dtype=np.float32),
        "other": np.zeros_like(vocal_residue, dtype=np.float32),
    }
    mix = write_wav(tmp_path / "mix.wav", drums + vocal_residue)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-post-debleedable", "{window_mix}"],
                        "post_minus_vocal_debleed": {
                            "strength": 1.0,
                            "low_hz": 700.0,
                            "high_hz": 1200.0,
                            "target_stems": ["drums"],
                            "passes": 1,
                        },
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        sf.write(stems_dir / "vocals.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)
        sf.write(stems_dir / "drums.flac", (drums + vocal_residue)[: len(window_audio)], mix_rate)
        for stem in ("bass", "guitar", "piano", "other"):
            sf.write(stems_dir / f"{stem}.flac", np.zeros_like(window_audio, dtype=np.float32), mix_rate)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["verdict"] == "auto_accept"
    assert candidate["post_minus_vocal_debleed"]["applied"] is True
    assert candidate["post_minus_vocal_debleed"]["passes"] == 1
    assert candidate["reconstruction"]["ok"] is True
    assert candidate["contamination_score"]["verdict"] == "auto_accept"


def test_vocal_projection_debleed_moves_correlated_vocal_leak_back_to_vocals(tmp_path: Path) -> None:
    source_audio = {
        "vocals": sine(880.0, seconds=0.5, amp=0.04),
        "drums": sine(110.0, seconds=0.5, amp=0.12),
        "bass": sine(220.0, seconds=0.5, amp=0.08),
        "guitar": sine(330.0, seconds=0.5, amp=0.06),
        "piano": sine(440.0, seconds=0.5, amp=0.05),
        "other": sine(550.0, seconds=0.5, amp=0.03),
    }
    drum_leak = 0.3 * source_audio["vocals"]
    guitar_leak = 0.2 * source_audio["vocals"]
    mix = write_wav(tmp_path / "mix.wav", sum(source_audio.values()))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "stem_search_recipe.v1",
                "candidates": [
                    {
                        "candidate_id": "c001",
                        "command_template": ["fixture-write-projection-debleed", "{window_mix}"],
                        "vocal_projection_debleed": {
                            "strength": 1.0,
                            "target_stems": ["drums", "guitar"],
                        },
                        "foreign_references": write_full_foreign_references(tmp_path, source_audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        window_audio, mix_rate = sf.read(command[1], dtype="float32", always_2d=False)
        stems_dir = fake_runner.stems_dir
        sf.write(stems_dir / "vocals.flac", (source_audio["vocals"] - drum_leak - guitar_leak)[: len(window_audio)], mix_rate)
        sf.write(stems_dir / "drums.flac", (source_audio["drums"] + drum_leak)[: len(window_audio)], mix_rate)
        sf.write(stems_dir / "bass.flac", source_audio["bass"][: len(window_audio)], mix_rate)
        sf.write(stems_dir / "guitar.flac", (source_audio["guitar"] + guitar_leak)[: len(window_audio)], mix_rate)
        sf.write(stems_dir / "piano.flac", source_audio["piano"][: len(window_audio)], mix_rate)
        sf.write(stems_dir / "other.flac", source_audio["other"][: len(window_audio)], mix_rate)

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert result["verdict"] == "auto_accept"
    assert candidate["vocal_projection_debleed"]["applied"] is True
    assert candidate["vocal_projection_debleed"]["per_stem"]["drums"]["projection_coeff"] == pytest.approx(0.6, abs=0.03)
    assert candidate["vocal_projection_debleed"]["per_stem"]["guitar"]["projection_coeff"] == pytest.approx(0.4, abs=0.03)
    assert candidate["reconstruction"]["ok"] is True


def test_recipe_candidate_missing_stem_blocks_candidate(tmp_path: Path) -> None:
    mix = write_wav(tmp_path / "mix.wav", sine(220.0, seconds=0.25))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps({"candidates": [{"candidate_id": "c001", "command_template": ["fixture-write-incomplete"]}]}),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        stems_dir = fake_runner.stems_dir
        for stem in ("vocals", "drums"):
            write_wav(stems_dir / f"{stem}.flac", sine(330.0, seconds=0.25, amp=0.02))

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert candidate["verdict"] == "blocked_missing_stem"
    assert "bass" in candidate["missing_stems"]
    assert result["verdict"] == "blocked_no_clean_candidate"


def test_bad_reconstruction_blocks_candidate(tmp_path: Path) -> None:
    mix = write_wav(tmp_path / "mix.wav", sine(220.0, seconds=0.25))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps({"candidates": [{"candidate_id": "c001", "command_template": ["fixture-write-bad-recon"]}]}),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], timeout_sec: int) -> None:
        stems_dir = fake_runner.stems_dir
        for stem in STEMS:
            write_wav(stems_dir / f"{stem}.flac", np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32))

    result = run_fully_automatic_stem_search(
        mix_path=mix,
        output_dir=tmp_path / "scratch" / "t001",
        recipe_path=recipe,
        window_sec=0.5,
        max_candidates=10,
        require_gpu=False,
        no_artifact_rag=True,
        command_runner=fake_runner,
    )

    candidate = result["candidates"][0]
    assert candidate["verdict"] == "blocked_bad_reconstruction"
    assert candidate["reconstruction"]["ok"] is False
    assert result["verdict"] == "blocked_no_clean_candidate"
