from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.no_reference_candidate_job_registry import CandidateJobConfig
from tools import run_no_reference_candidate_wave as wave
from tools.run_no_reference_candidate_wave import main, run_candidate_wave


SAMPLE_RATE = 16_000


def sine(freq: float, seconds: float, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return str(path)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def unit_job_config(config_id: str, target_mode: str) -> CandidateJobConfig:
    return CandidateJobConfig(
        candidate_config_id=config_id,
        candidate_family=f"{target_mode}_focused",
        target_mode=target_mode,
        source_model_id=f"unit-{target_mode}",
        source_tool="unit-existing-separator",
        entrypoint_type="python_script",
        command_template=("unit-runner", "{input_audio}", "{output_dir}"),
        expected_outputs={
            "vocal_path": "{output_dir}/vocals.wav" if target_mode == "vocal" else "",
            "minus_path": "{output_dir}/minus.wav" if target_mode == "minus" else "",
        },
        gpu_required=False,
        timeout_sec=5,
        max_retries=0,
        profile_allowed=("safe",),
        safety_notes=("unit wave config",),
    )


def shared_job_config(
    config_id: str,
    *,
    target_mode: str,
    family: str,
    group_id: str,
    produces_vocal: bool,
    produces_minus: bool,
    materialized_modes: tuple[str, ...],
) -> CandidateJobConfig:
    return CandidateJobConfig(
        candidate_config_id=config_id,
        candidate_family=family,
        target_mode=target_mode,
        source_model_id="unit-shared-model",
        source_tool="unit-existing-separator",
        entrypoint_type="python_script",
        command_template=("unit-runner", "{input_audio}", "{output_dir}"),
        expected_outputs={
            "vocal_path": "{output_dir}/vocals.wav",
            "minus_path": "{output_dir}/minus.wav",
        },
        gpu_required=False,
        timeout_sec=5,
        max_retries=0,
        profile_allowed=("safe", "balanced"),
        safety_notes=("unit shared wave config",),
        supported_parameters=("target_stem",),
        target_stem="minus" if produces_minus else "vocal",
        shared_execution_group_id=group_id,
        produces_vocal=produces_vocal,
        produces_minus=produces_minus,
        materialized_candidate_modes=materialized_modes,
        reuse_outputs_across_modes=True,
    )


def test_candidate_wave_creates_pool_from_mocked_separator_outputs(tmp_path: Path) -> None:
    input_audio = write_wav(tmp_path / "input" / "song.wav", sine(220, 1.5))
    source_vocal = write_wav(tmp_path / "mocked_separator" / "vocals.wav", sine(440, 1.5))
    source_minus = write_wav(tmp_path / "mocked_separator" / "minus.wav", sine(120, 1.5))
    source_jsonl = write_jsonl(
        tmp_path / "sources.jsonl",
        [
            {
                "candidate_config_id": "safe-vocal",
                "candidate_family": "vocal_focused",
                "source_model_id": "local-existing-vocal",
                "source_tool": "mocked_existing_separator",
                "target_mode": "vocal",
                "vocal_path": source_vocal,
            },
            {
                "candidate_config_id": "safe-minus",
                "candidate_family": "minus_focused",
                "source_model_id": "local-existing-minus",
                "source_tool": "mocked_existing_separator",
                "target_mode": "minus",
                "minus_path": source_minus,
            },
            {
                "candidate_config_id": "missing-minus",
                "candidate_family": "minus_focused",
                "source_tool": "mocked_existing_separator",
                "target_mode": "minus",
                "minus_path": str(tmp_path / "missing.wav"),
            },
        ],
    )
    output_dir = tmp_path / "candidate_wave"
    candidate_pool = tmp_path / "candidate_pool.jsonl"

    exit_code = main(
        [
            "--input-audio",
            input_audio,
            "--output-dir",
            str(output_dir),
            "--candidate-pool-jsonl",
            str(candidate_pool),
            "--profile",
            "safe",
            "--source-jsonl",
            str(source_jsonl),
            "--max-candidates",
            "6",
        ]
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in candidate_pool.read_text(encoding="utf-8").splitlines()]
    failures = [json.loads(line) for line in (output_dir / "candidate_wave_failures.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((output_dir / "candidate_wave_summary.json").read_text(encoding="utf-8"))

    assert len(rows) == 1
    row = rows[0]
    assert row["target_mode"] == "both"
    assert row["candidate_family"] == "vocal_focused+minus_focused"
    assert row["candidate_config_id"] == "safe-vocal+safe-minus"
    assert row["generation_profile"] == "safe"
    assert row["generation_status"] == "generated"
    assert row["can_use_for_vocal"] is True
    assert row["can_use_for_minus"] is True
    assert row["can_use_for_pair"] is True
    assert row["audio_rendered"] is False
    assert row["release_ready"] is False
    assert output_dir in Path(row["vocal_path"]).parents
    assert output_dir in Path(row["minus_path"]).parents
    assert Path(row["vocal_path"]).exists()
    assert Path(row["minus_path"]).exists()
    assert failures and failures[0]["skip_reason"] == "missing_minus_path"
    assert summary["candidates_generated"] == 1
    assert summary["complete_pairs"] == 1
    assert summary["failed_candidates"] == 1


def test_candidate_wave_is_deterministic_for_same_input_config_profile(tmp_path: Path) -> None:
    input_audio = write_wav(tmp_path / "input" / "song.wav", sine(220, 1.0))
    source_vocal = write_wav(tmp_path / "mocked_separator" / "vocals.wav", sine(440, 1.0))
    source_minus = write_wav(tmp_path / "mocked_separator" / "minus.wav", sine(120, 1.0))
    source_jsonl = write_jsonl(
        tmp_path / "sources.jsonl",
        [
            {"candidate_config_id": "v1", "candidate_family": "vocal_focused", "target_mode": "vocal", "vocal_path": source_vocal},
            {"candidate_config_id": "m1", "candidate_family": "minus_focused", "target_mode": "minus", "minus_path": source_minus},
        ],
    )

    first = run_candidate_wave(
        input_audio=Path(input_audio),
        output_dir=tmp_path / "run-a",
        candidate_pool_jsonl=tmp_path / "run-a" / "candidate_pool.jsonl",
        profile="safe",
        source_jsonl=source_jsonl,
        max_candidates=6,
    )
    second = run_candidate_wave(
        input_audio=Path(input_audio),
        output_dir=tmp_path / "run-b",
        candidate_pool_jsonl=tmp_path / "run-b" / "candidate_pool.jsonl",
        profile="safe",
        source_jsonl=source_jsonl,
        max_candidates=6,
    )

    assert [row["candidate_id"] for row in first["rows"]] == [row["candidate_id"] for row in second["rows"]]
    assert first["summary"]["release_ready_count"] == 0
    assert second["summary"]["release_ready_count"] == 0


def test_candidate_wave_run_jobs_creates_pool_rows_from_mocked_job_outputs(tmp_path: Path, monkeypatch) -> None:
    input_audio = write_wav(tmp_path / "input" / "song.wav", sine(220, 1.0))
    configs = [unit_job_config("unit-vocal", "vocal"), unit_job_config("unit-minus", "minus")]

    def fake_registry(profile: str | None = None):
        return configs

    def fake_job(**kwargs):
        config = next(config for config in configs if config.candidate_config_id == kwargs["candidate_config_id"])
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        vocal_path = ""
        minus_path = ""
        if config.target_mode == "vocal":
            vocal_path = write_wav(output_dir / "vocals.wav", sine(440, 1.0))
        if config.target_mode == "minus":
            minus_path = write_wav(output_dir / "minus.wav", sine(120, 1.0))
        manifest = {
            "candidate_config_id": config.candidate_config_id,
            "candidate_family": config.candidate_family,
            "target_mode": config.target_mode,
            "source_model_id": config.source_model_id,
            "source_tool": config.source_tool,
            "source_config_hash": config.source_config_hash,
            "generation_status": "generated",
            "skip_reason": "",
            "output_paths": {"vocal_path": vocal_path, "minus_path": minus_path},
            "job_manifest_path": str(kwargs["job_manifest_json"]),
            "job_progress_path": str(kwargs["job_progress_jsonl"]),
            "audio_rendered": True,
            "release_ready": False,
        }
        kwargs["job_manifest_json"].write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        kwargs["job_progress_jsonl"].write_text(json.dumps({"phase": "complete"}) + "\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(wave, "build_candidate_job_registry", fake_registry)
    monkeypatch.setattr(wave, "run_local_separator_candidate_job", fake_job)

    result = run_candidate_wave(
        input_audio=Path(input_audio),
        output_dir=tmp_path / "wave",
        candidate_pool_jsonl=tmp_path / "wave" / "candidate_pool.jsonl",
        profile="safe",
        execution_mode="run_jobs",
        max_candidates=4,
        max_runtime_sec=60,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "wave" / "candidate_pool.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    job_rows = [
        json.loads(line)
        for line in (tmp_path / "wave" / "candidate_wave_jobs.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert result["summary"]["jobs_attempted"] == 2
    assert result["summary"]["jobs_generated"] == 2
    assert result["summary"]["candidates_generated"] == 1
    assert rows and rows[0]["target_mode"] == "both"
    assert rows[0]["job_manifest_path"]
    assert rows[0]["job_progress_path"]
    assert rows[0]["audio_rendered"] is True
    assert rows[0]["release_ready"] is False
    assert len(job_rows) == 2


def test_candidate_wave_uses_normalized_job_paths_and_preserves_raw_provenance(tmp_path: Path, monkeypatch) -> None:
    input_audio = write_wav(tmp_path / "input" / "song.wav", sine(220, 1.0))
    configs = [unit_job_config("unit-vocal", "vocal"), unit_job_config("unit-minus", "minus")]
    captured: list[bool] = []

    def fake_registry(profile: str | None = None):
        return configs

    def fake_job(**kwargs):
        captured.append(kwargs["allow_shape_normalization"])
        config = next(config for config in configs if config.candidate_config_id == kwargs["candidate_config_id"])
        output_dir = kwargs["output_dir"]
        raw_dir = output_dir / "raw"
        normalized_dir = output_dir / "normalized"
        output_dir.mkdir(parents=True, exist_ok=True)
        vocal_path = ""
        minus_path = ""
        raw_vocal_path = ""
        raw_minus_path = ""
        normalized_vocal_path = ""
        normalized_minus_path = ""
        if config.target_mode == "vocal":
            raw_vocal_path = write_wav(raw_dir / "vocals.wav", sine(440, 1.0))
            normalized_vocal_path = write_wav(normalized_dir / "vocals.wav", sine(440, 1.0))
            vocal_path = normalized_vocal_path
        if config.target_mode == "minus":
            raw_minus_path = write_wav(raw_dir / "minus.wav", sine(120, 1.0))
            normalized_minus_path = write_wav(normalized_dir / "minus.wav", sine(120, 1.0))
            minus_path = normalized_minus_path
        manifest = {
            "candidate_config_id": config.candidate_config_id,
            "candidate_family": config.candidate_family,
            "target_mode": config.target_mode,
            "source_model_id": config.source_model_id,
            "source_tool": config.source_tool,
            "source_config_hash": config.source_config_hash,
            "generation_status": "generated_with_shape_normalization",
            "skip_reason": "",
            "output_paths": {"vocal_path": vocal_path, "minus_path": minus_path},
            "raw_vocal_path": raw_vocal_path,
            "raw_minus_path": raw_minus_path,
            "normalized_vocal_path": normalized_vocal_path,
            "normalized_minus_path": normalized_minus_path,
            "shape_normalization_applied": True,
            "shape_normalization_reason": "sample_rate_mismatch",
            "job_manifest_path": str(kwargs["job_manifest_json"]),
            "job_progress_path": str(kwargs["job_progress_jsonl"]),
            "audio_rendered": True,
            "release_ready": False,
        }
        kwargs["job_manifest_json"].write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        kwargs["job_progress_jsonl"].write_text(json.dumps({"phase": "complete"}) + "\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(wave, "build_candidate_job_registry", fake_registry)
    monkeypatch.setattr(wave, "run_local_separator_candidate_job", fake_job)

    result = run_candidate_wave(
        input_audio=Path(input_audio),
        output_dir=tmp_path / "wave",
        candidate_pool_jsonl=tmp_path / "wave" / "candidate_pool.jsonl",
        profile="safe",
        execution_mode="run_jobs",
        max_candidates=4,
        max_runtime_sec=60,
        allow_shape_normalization=True,
        candidate_diversity_profile="safe",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "wave" / "candidate_pool.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert captured == [True, True]
    assert result["summary"]["jobs_generated"] == 2
    assert result["summary"]["jobs_generated_raw"] == 0
    assert result["summary"]["jobs_generated_with_shape_normalization"] == 2
    assert result["summary"]["jobs_failed_shape_unrecoverable"] == 0
    assert result["summary"]["candidate_families_generated"]
    assert result["summary"]["target_modes_generated"] == ["both"]
    assert rows and rows[0]["target_mode"] == "both"
    assert "/normalized/" in rows[0]["vocal_path"].replace("\\", "/")
    assert "/normalized/" in rows[0]["minus_path"].replace("\\", "/")
    assert "/raw/" in rows[0]["raw_vocal_path"].replace("\\", "/")
    assert "/raw/" in rows[0]["raw_minus_path"].replace("\\", "/")
    assert rows[0]["shape_normalization_applied"] is True
    assert rows[0]["route_id"]
    assert "route_family" in rows[0]
    assert rows[0]["can_run_without_download"] is True
    assert rows[0]["can_run_without_registry_edit"] is True
    assert rows[0]["release_ready"] is False


def test_job_manifest_to_asset_preserves_direct_route_provenance(tmp_path: Path) -> None:
    input_audio = write_wav(tmp_path / "input" / "song.wav", sine(220, 1.0))
    vocal_path = write_wav(tmp_path / "job" / "vocals.wav", sine(440, 1.0))
    minus_path = write_wav(tmp_path / "job" / "minus.wav", sine(120, 1.0))

    asset, failure = wave.job_manifest_to_asset(
        run_id="run-a",
        track_id="track-a",
        input_audio=Path(input_audio),
        profile="hard_case",
        manifest={
            "candidate_config_id": "local_registered__htdemucs_6s__both__shared",
            "route_id": "route-htdemucs-6s",
            "route_family": "htdemucs_6s",
            "candidate_family": "artifact_safe_default",
            "target_mode": "both",
            "source_model_id": "htdemucs_6s",
            "source_tool": "tools/run_registered_model_candidate.py",
            "source_config_hash": "hash-a",
            "generation_status": "generated",
            "output_paths": {"vocal_path": vocal_path, "minus_path": minus_path},
            "can_run_without_download": True,
            "can_run_without_registry_edit": True,
            "forbidden_reason": "",
            "release_ready": False,
            "audio_rendered": True,
        },
    )

    assert failure is None
    assert asset is not None
    assert asset["route_id"] == "route-htdemucs-6s"
    assert asset["route_family"] == "htdemucs_6s"
    assert asset["can_run_without_download"] is True
    assert asset["can_run_without_registry_edit"] is True
    assert asset["forbidden_reason"] == ""


def test_materialize_asset_adds_default_route_provenance_for_source_both_rows(tmp_path: Path) -> None:
    input_audio = write_wav(tmp_path / "input" / "song.wav", sine(220, 1.0))
    vocal_path = write_wav(tmp_path / "source" / "vocals.wav", sine(440, 1.0))
    minus_path = write_wav(tmp_path / "source" / "minus.wav", sine(120, 1.0))

    asset, failure = wave.materialize_asset(
        run_id="run-a",
        track_id="track-a",
        input_audio=Path(input_audio),
        output_dir=tmp_path / "out",
        row={
            "candidate_config_id": "source-both",
            "candidate_family": "source_pair",
            "target_mode": "both",
            "vocal_path": vocal_path,
            "minus_path": minus_path,
            "source_tool": "existing_source_pool",
            "source_model_id": "source-model",
        },
        profile="safe",
    )

    assert failure is None
    assert asset is not None
    assert asset["route_id"] == "source-both"
    assert "route_family" in asset
    assert asset["can_run_without_download"] is True
    assert asset["can_run_without_registry_edit"] is True
    assert asset["forbidden_reason"] == ""


def test_candidate_wave_reuses_shared_execution_and_materializes_modes(tmp_path: Path, monkeypatch) -> None:
    input_audio = write_wav(tmp_path / "input" / "song.wav", sine(220, 1.0))
    configs = [
        shared_job_config("unit-vocal", target_mode="vocal", family="vocal_focused_default", group_id="shared-a", produces_vocal=True, produces_minus=False, materialized_modes=("vocal",)),
        shared_job_config("unit-minus", target_mode="minus", family="minus_focused_default", group_id="shared-a", produces_vocal=False, produces_minus=True, materialized_modes=("minus",)),
        shared_job_config("unit-both", target_mode="both", family="balanced_pair_default", group_id="shared-a", produces_vocal=True, produces_minus=True, materialized_modes=("vocal", "minus", "both")),
    ]
    calls: list[str] = []

    def fake_registry(profile: str | None = None):
        return configs

    def fake_job(**kwargs):
        calls.append(kwargs["candidate_config_id"])
        assert kwargs["candidate_config_id"] == "unit-both"
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        vocal_path = write_wav(output_dir / "vocals.wav", sine(440, 1.0))
        minus_path = write_wav(output_dir / "minus.wav", sine(120, 1.0))
        manifest = {
            "candidate_config_id": "unit-both",
            "candidate_family": "balanced_pair_default",
            "target_mode": "both",
            "source_model_id": "unit-shared-model",
            "source_tool": "unit-existing-separator",
            "source_config_hash": configs[2].source_config_hash,
            "generation_status": "generated",
            "skip_reason": "",
            "output_paths": {"vocal_path": vocal_path, "minus_path": minus_path},
            "raw_vocal_path": vocal_path,
            "raw_minus_path": minus_path,
            "normalized_vocal_path": vocal_path,
            "normalized_minus_path": minus_path,
            "shape_normalization_applied": False,
            "shape_normalization_reason": "shape_already_valid",
            "job_manifest_path": str(kwargs["job_manifest_json"]),
            "job_progress_path": str(kwargs["job_progress_jsonl"]),
            "audio_rendered": True,
            "release_ready": False,
        }
        kwargs["job_manifest_json"].write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        kwargs["job_progress_jsonl"].write_text(json.dumps({"phase": "complete"}) + "\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(wave, "build_candidate_job_registry", fake_registry)
    monkeypatch.setattr(wave, "run_local_separator_candidate_job", fake_job)

    result = run_candidate_wave(
        input_audio=Path(input_audio),
        output_dir=tmp_path / "wave",
        candidate_pool_jsonl=tmp_path / "wave" / "candidate_pool.jsonl",
        profile="safe",
        execution_mode="run_jobs",
        max_candidates=6,
        max_runtime_sec=60,
    )

    pool_rows = [
        json.loads(line)
        for line in (tmp_path / "wave" / "candidate_pool.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    materialized_rows = [
        json.loads(line)
        for line in (tmp_path / "wave" / "candidate_wave_materialized_assets.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert calls == ["unit-both"]
    assert result["summary"]["jobs_attempted"] == 1
    assert result["summary"]["duplicate_jobs_avoided"] == 2
    assert len(pool_rows) == 1
    assert pool_rows[0]["target_mode"] == "both"
    assert {row["target_mode"] for row in materialized_rows} == {"vocal", "minus", "both"}
    assert len({row["candidate_id"] for row in materialized_rows}) == 3
    assert len({row["job_manifest_path"] for row in materialized_rows}) == 1
    assert all(row["release_ready"] is False for row in materialized_rows)
    persisted_job_manifest = json.loads(
        Path(materialized_rows[0]["job_manifest_path"]).read_text(encoding="utf-8")
    )
    assert persisted_job_manifest["job_plan_id"]
    assert persisted_job_manifest["shared_execution_group_id"] == "shared-a"
    assert persisted_job_manifest["materialized_candidate_modes"] == ["vocal", "minus", "both"]


def test_candidate_wave_runs_only_route_filtered_hard_case_configs(tmp_path: Path, monkeypatch) -> None:
    input_audio = write_wav(tmp_path / "input" / "song.wav", sine(220, 1.0))
    configs = [
        shared_job_config("unit-both", target_mode="both", family="balanced_pair_default", group_id="shared-a", produces_vocal=True, produces_minus=True, materialized_modes=("vocal", "minus", "both")),
        shared_job_config("unit-minus-a", target_mode="minus", family="minus_focused_existing_model", group_id="shared-b", produces_vocal=False, produces_minus=True, materialized_modes=("minus",)),
        shared_job_config("unit-minus-b", target_mode="minus", family="minus_focused_existing_model", group_id="shared-c", produces_vocal=False, produces_minus=True, materialized_modes=("minus",)),
    ]
    configs[1] = CandidateJobConfig(**{**configs[1].to_manifest(), "hard_case_routes": ["generate_minus_focused_variants"]})
    configs[2] = CandidateJobConfig(**{**configs[2].to_manifest(), "hard_case_routes": ["generate_minus_focused_variants"]})
    calls: list[str] = []

    def fake_registry(profile: str | None = None):
        return configs

    def fake_job(**kwargs):
        calls.append(kwargs["candidate_config_id"])
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        minus_path = write_wav(output_dir / "minus.wav", sine(120, 1.0))
        manifest = {
            "candidate_config_id": kwargs["candidate_config_id"],
            "candidate_family": "minus_focused_existing_model",
            "target_mode": "minus",
            "source_model_id": "unit-model",
            "source_tool": "unit-tool",
            "source_config_hash": configs[1].source_config_hash,
            "generation_status": "generated",
            "skip_reason": "",
            "output_paths": {"vocal_path": "", "minus_path": minus_path},
            "raw_minus_path": minus_path,
            "normalized_minus_path": minus_path,
            "shape_normalization_applied": False,
            "shape_normalization_reason": "shape_already_valid",
            "job_manifest_path": str(kwargs["job_manifest_json"]),
            "job_progress_path": str(kwargs["job_progress_jsonl"]),
            "audio_rendered": True,
            "release_ready": False,
        }
        kwargs["job_manifest_json"].write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        kwargs["job_progress_jsonl"].write_text(json.dumps({"phase": "complete"}) + "\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(wave, "build_candidate_job_registry", fake_registry)
    monkeypatch.setattr(wave, "run_local_separator_candidate_job", fake_job)

    result = run_candidate_wave(
        input_audio=Path(input_audio),
        output_dir=tmp_path / "wave",
        candidate_pool_jsonl=tmp_path / "wave" / "candidate_pool.jsonl",
        profile="balanced",
        execution_mode="run_jobs",
        max_candidates=4,
        max_runtime_sec=60,
        candidate_diversity_profile="balanced",
        candidate_route_filter="generate_minus_focused_variants",
        max_job_count=1,
    )

    plan_rows = [
        json.loads(line)
        for line in (tmp_path / "wave" / "candidate_job_plan.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert calls == ["unit-minus-a"]
    assert result["summary"]["candidate_route_filter"] == "generate_minus_focused_variants"
    assert result["summary"]["max_job_count"] == 1
    assert any(row["reason_skipped"] == "route_not_selected" for row in plan_rows)
    assert any(row["reason_skipped"] == "max_candidates_limit" for row in plan_rows)


def test_candidate_wave_route_filter_accepts_exact_route_id(tmp_path: Path, monkeypatch) -> None:
    input_audio = write_wav(tmp_path / "input" / "song.wav", sine(220, 1.0))
    target = shared_job_config("unit-both-a", target_mode="both", family="balanced_pair_default", group_id="shared-a", produces_vocal=True, produces_minus=True, materialized_modes=("vocal", "minus", "both"))
    skipped = shared_job_config("unit-both-b", target_mode="both", family="balanced_pair_default", group_id="shared-b", produces_vocal=True, produces_minus=True, materialized_modes=("vocal", "minus", "both"))
    configs = [
        CandidateJobConfig(**{**target.to_manifest(), "route_id": "local_registered__htdemucs_6s__both__shared", "hard_case_routes": []}),
        CandidateJobConfig(**{**skipped.to_manifest(), "route_id": "local_registered__htdemucs_ft__both__shared", "hard_case_routes": []}),
    ]
    calls: list[str] = []

    def fake_registry(profile: str | None = None):
        return configs

    def fake_job(**kwargs):
        calls.append(kwargs["candidate_config_id"])
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        vocal_path = write_wav(output_dir / "vocals.wav", sine(440, 1.0))
        minus_path = write_wav(output_dir / "minus.wav", sine(120, 1.0))
        manifest = {
            "candidate_config_id": kwargs["candidate_config_id"],
            "route_id": "local_registered__htdemucs_6s__both__shared",
            "route_family": "htdemucs_6s",
            "candidate_family": "balanced_pair_default",
            "target_mode": "both",
            "source_model_id": "htdemucs_6s",
            "source_tool": "unit-tool",
            "source_config_hash": configs[0].source_config_hash,
            "generation_status": "generated",
            "skip_reason": "",
            "output_paths": {"vocal_path": vocal_path, "minus_path": minus_path},
            "job_manifest_path": str(kwargs["job_manifest_json"]),
            "job_progress_path": str(kwargs["job_progress_jsonl"]),
            "audio_rendered": True,
            "release_ready": False,
        }
        kwargs["job_manifest_json"].write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        kwargs["job_progress_jsonl"].write_text(json.dumps({"phase": "complete"}) + "\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(wave, "build_candidate_job_registry", fake_registry)
    monkeypatch.setattr(wave, "run_local_separator_candidate_job", fake_job)

    result = run_candidate_wave(
        input_audio=Path(input_audio),
        output_dir=tmp_path / "wave",
        candidate_pool_jsonl=tmp_path / "wave" / "candidate_pool.jsonl",
        profile="balanced",
        execution_mode="run_jobs",
        max_candidates=4,
        max_runtime_sec=60,
        candidate_diversity_profile="balanced",
        candidate_route_filter="local_registered__htdemucs_6s__both__shared",
    )

    assert calls == ["unit-both-a"]
    assert result["summary"]["jobs_attempted"] == 1
    assert result["summary"]["candidate_route_filter"] == "local_registered__htdemucs_6s__both__shared"


def test_candidate_wave_auto_falls_back_to_run_jobs_when_no_existing_outputs(tmp_path: Path, monkeypatch) -> None:
    input_audio = write_wav(tmp_path / "input" / "song.wav", sine(220, 1.0))
    configs = [unit_job_config("unit-vocal", "vocal"), unit_job_config("unit-minus", "minus")]
    calls: list[str] = []

    def fake_registry(profile: str | None = None):
        return configs

    def fake_job(**kwargs):
        calls.append(str(kwargs["candidate_config_id"]))
        config = next(config for config in configs if config.candidate_config_id == kwargs["candidate_config_id"])
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        vocal_path = write_wav(output_dir / "vocals.wav", sine(440, 1.0)) if config.target_mode == "vocal" else ""
        minus_path = write_wav(output_dir / "minus.wav", sine(120, 1.0)) if config.target_mode == "minus" else ""
        kwargs["job_manifest_json"].write_text("{}", encoding="utf-8")
        kwargs["job_progress_jsonl"].write_text(json.dumps({"phase": "complete"}) + "\n", encoding="utf-8")
        return {
            "candidate_config_id": config.candidate_config_id,
            "candidate_family": config.candidate_family,
            "target_mode": config.target_mode,
            "source_model_id": config.source_model_id,
            "source_tool": config.source_tool,
            "source_config_hash": config.source_config_hash,
            "generation_status": "generated",
            "skip_reason": "",
            "output_paths": {"vocal_path": vocal_path, "minus_path": minus_path},
            "job_manifest_path": str(kwargs["job_manifest_json"]),
            "job_progress_path": str(kwargs["job_progress_jsonl"]),
            "audio_rendered": True,
            "release_ready": False,
        }

    monkeypatch.setattr(wave, "build_candidate_job_registry", fake_registry)
    monkeypatch.setattr(wave, "run_local_separator_candidate_job", fake_job)

    result = run_candidate_wave(
        input_audio=Path(input_audio),
        output_dir=tmp_path / "wave",
        candidate_pool_jsonl=tmp_path / "wave" / "candidate_pool.jsonl",
        profile="safe",
        execution_mode="auto",
        max_candidates=4,
        max_runtime_sec=60,
    )

    assert calls == ["unit-minus", "unit-vocal"]
    assert result["summary"]["candidate_execution_mode"] == "auto"
    assert result["summary"]["jobs_attempted"] == 2
    assert result["summary"]["complete_pairs"] == 1


def test_candidate_wave_records_partial_job_failures(tmp_path: Path, monkeypatch) -> None:
    input_audio = write_wav(tmp_path / "input" / "song.wav", sine(220, 1.0))
    configs = [unit_job_config("unit-vocal", "vocal"), unit_job_config("unit-minus", "minus")]

    def fake_registry(profile: str | None = None):
        return configs

    def fake_job(**kwargs):
        config = next(config for config in configs if config.candidate_config_id == kwargs["candidate_config_id"])
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        if config.target_mode == "minus":
            return {
                "candidate_config_id": config.candidate_config_id,
                "candidate_family": config.candidate_family,
                "target_mode": config.target_mode,
                "source_model_id": config.source_model_id,
                "source_tool": config.source_tool,
                "source_config_hash": config.source_config_hash,
                "generation_status": "failed_missing_output",
                "skip_reason": "failed_missing_output",
                "output_paths": {"vocal_path": "", "minus_path": ""},
                "job_manifest_path": str(kwargs["job_manifest_json"]),
                "job_progress_path": str(kwargs["job_progress_jsonl"]),
                "audio_rendered": False,
                "release_ready": False,
            }
        vocal_path = write_wav(output_dir / "vocals.wav", sine(440, 1.0))
        return {
            "candidate_config_id": config.candidate_config_id,
            "candidate_family": config.candidate_family,
            "target_mode": config.target_mode,
            "source_model_id": config.source_model_id,
            "source_tool": config.source_tool,
            "source_config_hash": config.source_config_hash,
            "generation_status": "generated",
            "skip_reason": "",
            "output_paths": {"vocal_path": vocal_path, "minus_path": ""},
            "job_manifest_path": str(kwargs["job_manifest_json"]),
            "job_progress_path": str(kwargs["job_progress_jsonl"]),
            "audio_rendered": True,
            "release_ready": False,
        }

    monkeypatch.setattr(wave, "build_candidate_job_registry", fake_registry)
    monkeypatch.setattr(wave, "run_local_separator_candidate_job", fake_job)

    result = run_candidate_wave(
        input_audio=Path(input_audio),
        output_dir=tmp_path / "wave",
        candidate_pool_jsonl=tmp_path / "wave" / "candidate_pool.jsonl",
        profile="safe",
        execution_mode="run_jobs",
        max_candidates=4,
        max_runtime_sec=60,
        continue_on_failure=True,
    )

    failure_rows = [
        json.loads(line)
        for line in (tmp_path / "wave" / "candidate_wave_failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert result["summary"]["jobs_attempted"] == 2
    assert result["summary"]["jobs_failed"] == 1
    assert result["summary"]["complete_pairs"] == 0
    assert any(row["skip_reason"] == "failed_missing_output" for row in failure_rows)
