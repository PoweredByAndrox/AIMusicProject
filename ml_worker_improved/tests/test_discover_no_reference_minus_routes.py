from __future__ import annotations

import json
from pathlib import Path

from tools.no_reference_candidate_job_registry import CandidateJobConfig
from worker.model_registry import ModelRegistry, ModelRegistryEntry


def minus_config(
    config_id: str,
    *,
    model_id: str = "local_minus_model",
    entrypoint_type: str = "python_script",
    available: bool = True,
    unavailable_reason: str = "",
    family: str = "minus_focused_existing_model",
) -> CandidateJobConfig:
    return CandidateJobConfig(
        candidate_config_id=config_id,
        candidate_family=family,
        target_mode="minus",
        source_model_id=model_id,
        source_tool="tools/run_registered_model_candidate.py",
        entrypoint_type=entrypoint_type,
        command_template=("python", "/app/tools/run_registered_model_candidate.py"),
        expected_outputs={"vocal_path": "", "minus_path": "{output_dir}/minus.wav"},
        gpu_required=True,
        timeout_sec=100,
        max_retries=0,
        profile_allowed=("safe", "balanced", "exploration"),
        safety_notes=("uses_existing_model_registry_entry", "no_model_download"),
        supported_parameters=("source_audio_path", "output_dir", "model_id", "device", "target_stem"),
        target_stem="minus",
        produces_minus=True,
        available=available,
        unavailable_reason=unavailable_reason,
    )


def registry_with_entry(
    tmp_path: Path,
    *,
    model_id: str = "local_minus_model",
    required_path: str = "models/local_minus.ckpt",
    target_stems: tuple[str, ...] = ("instrumental",),
) -> ModelRegistry:
    return ModelRegistry(
        entries=(
            ModelRegistryEntry(
                model_id=model_id,
                display_name=model_id,
                family="demucs",
                target_stems=target_stems,
                output_stems=target_stems,
                command_template=("python", "/app/tools/export_hq_stems_full.py"),
                expected_manifest="{output_dir}/stem_manifest.json",
                required_paths=(required_path,),
            ),
        ),
        runtime_root=tmp_path,
    )


def test_available_local_route_is_emitted_with_download_and_registry_guards(tmp_path: Path) -> None:
    from tools.discover_no_reference_minus_routes import discover_minus_routes

    entrypoint = tmp_path / "tools" / "run_registered_model_candidate.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# local entrypoint\n", encoding="utf-8")
    model_file = tmp_path / "models" / "local_minus.ckpt"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("weights-present", encoding="utf-8")
    registry = registry_with_entry(tmp_path)

    manifest = discover_minus_routes(
        registry=registry,
        registry_configs=[minus_config("local-minus")],
        entrypoint_path=entrypoint,
    )

    assert manifest["summary"]["discovered_routes_total"] == 1
    row = manifest["routes"][0]
    assert row["route_id"]
    assert row["candidate_config_id"] == "local-minus"
    assert row["supports_minus"] is True
    assert row["entrypoint_available"] is True
    assert row["model_files_available"] is True
    assert row["available"] is True
    assert row["can_run_without_download"] is True
    assert row["can_run_without_registry_edit"] is True
    assert row["release_ready"] is False


def test_missing_model_files_mark_route_unavailable(tmp_path: Path) -> None:
    from tools.discover_no_reference_minus_routes import discover_minus_routes

    entrypoint = tmp_path / "tools" / "run_registered_model_candidate.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# local entrypoint\n", encoding="utf-8")
    registry = registry_with_entry(tmp_path)

    manifest = discover_minus_routes(
        registry=registry,
        registry_configs=[minus_config("missing-minus")],
        entrypoint_path=entrypoint,
    )

    row = manifest["routes"][0]
    assert row["available"] is False
    assert row["model_files_available"] is False
    assert "missing_path:models/local_minus.ckpt" in row["unavailable_reason"]


def test_unsupported_cli_parameter_marks_variant_unavailable(tmp_path: Path) -> None:
    from tools.discover_no_reference_minus_routes import discover_minus_routes

    entrypoint = tmp_path / "tools" / "run_registered_model_candidate.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# local entrypoint\n", encoding="utf-8")
    model_file = tmp_path / "models" / "local_minus.ckpt"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("weights-present", encoding="utf-8")
    registry = registry_with_entry(tmp_path)

    manifest = discover_minus_routes(
        registry=registry,
        registry_configs=[
            minus_config(
                "unsupported-overlap",
                entrypoint_type="unavailable",
                available=False,
                unavailable_reason="unsupported_parameter:overlap",
                family="high_overlap_variant",
            )
        ],
        entrypoint_path=entrypoint,
    )

    row = manifest["routes"][0]
    assert row["available"] is False
    assert row["forbidden_reason"] == "unsupported_cli_parameter:overlap"


def test_discovery_does_not_invent_missing_minus_route_or_mutate_registry(tmp_path: Path) -> None:
    from tools.discover_no_reference_minus_routes import discover_minus_routes

    registry = ModelRegistry(
        entries=(
            ModelRegistryEntry(
                model_id="vocal_only",
                display_name="vocal_only",
                family="mdx",
                target_stems=("vocals",),
                output_stems=("vocals",),
                command_template=("python", "vocal.py"),
                expected_manifest="{output_dir}/manifest.json",
            ),
        ),
        runtime_root=tmp_path,
    )
    before = json.dumps(registry.to_manifest(), sort_keys=True)

    manifest = discover_minus_routes(
        registry=registry,
        registry_configs=[],
        entrypoint_path=tmp_path / "missing.py",
    )

    assert manifest["routes"] == []
    assert json.dumps(registry.to_manifest(), sort_keys=True) == before
    assert manifest["summary"]["discovered_routes_total"] == 0


def test_registry_manifest_includes_stem2_route_fields() -> None:
    manifest = minus_config("route-fields").to_manifest()

    assert manifest["route_id"]
    assert manifest["route_family"]
    assert manifest["can_run_without_download"] is True
    assert manifest["can_run_without_registry_edit"] is True
    assert "forbidden_reason" in manifest


def test_no_cleanup_safety_note_does_not_mark_route_forbidden(tmp_path: Path) -> None:
    from tools.discover_no_reference_minus_routes import discover_minus_routes

    entrypoint = tmp_path / "tools" / "run_registered_model_candidate.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# local entrypoint\n", encoding="utf-8")
    model_file = tmp_path / "models" / "local_minus.ckpt"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("weights-present", encoding="utf-8")
    registry = registry_with_entry(tmp_path)
    config = minus_config("local-minus")
    config = CandidateJobConfig(
        **{
            **config.to_manifest(),
            "command_template": tuple(config.command_template),
            "profile_allowed": tuple(config.profile_allowed),
            "safety_notes": ("no_cleanup_chain", "no_model_download"),
            "supported_parameters": tuple(config.supported_parameters),
            "materialized_candidate_modes": tuple(config.materialized_candidate_modes),
            "hard_case_routes": tuple(config.hard_case_routes),
            "expected_outputs": dict(config.expected_outputs),
        }
    )

    manifest = discover_minus_routes(
        registry=registry,
        registry_configs=[config],
        entrypoint_path=entrypoint,
    )

    assert manifest["routes"][0]["forbidden_reason"] == ""
    assert manifest["routes"][0]["available"] is True
