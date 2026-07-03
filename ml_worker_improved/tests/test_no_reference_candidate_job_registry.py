from pathlib import Path

from tools.no_reference_candidate_job_registry import (
    STEM_SUM_MINUS_FORBIDDEN_REASON,
    build_candidate_job_registry,
    registered_model_command_template,
    registered_model_device,
)
from tools.no_reference_candidate_job_scheduler import build_candidate_job_plan
from worker.model_registry import ModelRegistry, ModelRegistryEntry


def registry_entry(model_id: str, *, output_stems: tuple[str, ...]) -> ModelRegistryEntry:
    return ModelRegistryEntry(
        model_id=model_id,
        display_name=model_id,
        family="unit",
        target_stems=output_stems,
        output_stems=output_stems,
        command_template=("python", "tool.py", "{source_audio_path}", "{output_dir}"),
        expected_manifest="{output_dir}/stem_manifest.json",
    )


def test_safe_minus_registry_uses_dedicated_instrumental_route_not_multistem_sum(tmp_path: Path) -> None:
    registry = ModelRegistry(
        entries=(
            registry_entry("htdemucs_ft", output_stems=("vocals", "drums", "bass", "other")),
            registry_entry(
                "htdemucs_6s",
                output_stems=("vocals", "drums", "bass", "other", "piano", "guitar"),
            ),
            registry_entry(
                "audio_separator_bs_roformer_viperx_1297_vendor_config_vocals",
                output_stems=("vocals", "instrumental"),
            ),
        )
    )

    configs = build_candidate_job_registry(profile="safe", registry=registry)
    by_id = {config.candidate_config_id: config for config in configs}

    dedicated = by_id[
        "local_registered__audio_separator_bs_roformer_viperx_1297_vendor_config_vocals__both__minus_direct"
    ]
    htdemucs_both = by_id["local_registered__htdemucs_ft__both__shared"]
    htdemucs_minus = by_id["local_registered__htdemucs_ft__minus"]

    assert dedicated.entrypoint_type == "python_script"
    assert dedicated.target_stem == "minus"
    assert dedicated.produces_minus is True
    assert dedicated.produces_both is True
    assert dedicated.forbidden_reason == ""

    for config in (htdemucs_both, htdemucs_minus):
        assert config.entrypoint_type == "unavailable"
        assert config.forbidden_reason == STEM_SUM_MINUS_FORBIDDEN_REASON
        assert STEM_SUM_MINUS_FORBIDDEN_REASON in config.unavailable_reason

    exploration_configs = build_candidate_job_registry(profile="exploration", registry=registry)
    for config in exploration_configs:
        if config.source_model_id.startswith("htdemucs") and config.target_stem == "minus":
            assert config.entrypoint_type == "unavailable"
            assert config.forbidden_reason == STEM_SUM_MINUS_FORBIDDEN_REASON
            assert STEM_SUM_MINUS_FORBIDDEN_REASON in config.unavailable_reason
        if config.source_model_id.startswith("htdemucs") and config.target_stem == "vocal":
            assert config.forbidden_reason != STEM_SUM_MINUS_FORBIDDEN_REASON

    plan = build_candidate_job_plan(
        input_audio_path=tmp_path / "song.wav",
        registry_configs=configs,
        candidate_profile="safe",
        candidate_diversity_profile="safe",
        max_candidates=2,
        max_runtime_sec=2400,
    )
    runnable = [row for row in plan if row["can_run"] is True]
    unavailable = {
        row["candidate_config_id"]: row
        for row in plan
        if row["reason_skipped"] == "unavailable_entrypoint"
    }

    assert runnable[0]["candidate_config_id"] == dedicated.candidate_config_id
    assert unavailable[htdemucs_both.candidate_config_id]["forbidden_reason"] == STEM_SUM_MINUS_FORBIDDEN_REASON
    assert unavailable[htdemucs_minus.candidate_config_id]["forbidden_reason"] == STEM_SUM_MINUS_FORBIDDEN_REASON

    balanced_configs = build_candidate_job_registry(profile="balanced", registry=registry)
    balanced_by_id = {config.candidate_config_id: config for config in balanced_configs}
    balanced_dedicated = balanced_by_id[dedicated.candidate_config_id]
    balanced_htdemucs_6s_both = balanced_by_id["local_registered__htdemucs_6s__both__shared"]
    balanced_htdemucs_6s_minus = balanced_by_id["local_registered__htdemucs_6s__minus"]

    balanced_plan = build_candidate_job_plan(
        input_audio_path=tmp_path / "song.wav",
        registry_configs=balanced_configs,
        candidate_profile="balanced",
        candidate_diversity_profile="balanced",
        max_candidates=2,
        max_runtime_sec=2400,
    )
    balanced_runnable = [row for row in balanced_plan if row["can_run"] is True]
    balanced_unavailable = {
        row["candidate_config_id"]: row
        for row in balanced_plan
        if row["reason_skipped"] == "unavailable_entrypoint"
    }

    assert balanced_runnable[0]["candidate_config_id"] == balanced_dedicated.candidate_config_id
    assert balanced_htdemucs_6s_both.entrypoint_type == "unavailable"
    assert balanced_htdemucs_6s_minus.entrypoint_type == "unavailable"
    assert (
        balanced_unavailable[balanced_htdemucs_6s_both.candidate_config_id]["forbidden_reason"]
        == STEM_SUM_MINUS_FORBIDDEN_REASON
    )
    assert (
        balanced_unavailable[balanced_htdemucs_6s_minus.candidate_config_id]["forbidden_reason"]
        == STEM_SUM_MINUS_FORBIDDEN_REASON
    )


def test_registered_model_command_template_keeps_stable_device_placeholder(monkeypatch) -> None:
    monkeypatch.setenv("AIMUSIC_REGISTERED_MODEL_DEVICE", "mps")
    monkeypatch.setenv("AIMUSIC_SEPARATOR_DEVICE", "cpu")

    template = registered_model_command_template()

    assert template[template.index("--device") + 1] == "{device}"


def test_registered_model_device_defaults_to_cuda(monkeypatch) -> None:
    monkeypatch.delenv("AIMUSIC_REGISTERED_MODEL_DEVICE", raising=False)
    monkeypatch.delenv("AIMUSIC_SEPARATOR_DEVICE", raising=False)

    assert registered_model_device() == "cuda"


def test_registered_model_device_can_be_overridden_for_mac(monkeypatch) -> None:
    monkeypatch.setenv("AIMUSIC_REGISTERED_MODEL_DEVICE", "mps")
    monkeypatch.setenv("AIMUSIC_SEPARATOR_DEVICE", "cuda")

    assert registered_model_device() == "mps"


def test_registered_model_device_falls_back_to_separator_device(monkeypatch) -> None:
    monkeypatch.delenv("AIMUSIC_REGISTERED_MODEL_DEVICE", raising=False)
    monkeypatch.setenv("AIMUSIC_SEPARATOR_DEVICE", "mps")

    assert registered_model_device() == "mps"


def test_registered_model_source_hash_is_stable_across_device_env(monkeypatch) -> None:
    registry = ModelRegistry(
        entries=(
            registry_entry(
                "audio_separator_bs_roformer_viperx_1297_vendor_config_vocals",
                output_stems=("vocals", "instrumental"),
            ),
        )
    )

    monkeypatch.delenv("AIMUSIC_REGISTERED_MODEL_DEVICE", raising=False)
    monkeypatch.delenv("AIMUSIC_SEPARATOR_DEVICE", raising=False)
    default_config = {
        config.candidate_config_id: config
        for config in build_candidate_job_registry(profile="safe", registry=registry)
    }["local_registered__audio_separator_bs_roformer_viperx_1297_vendor_config_vocals__both__minus_direct"]

    monkeypatch.setenv("AIMUSIC_REGISTERED_MODEL_DEVICE", "mps")
    overridden_config = {
        config.candidate_config_id: config
        for config in build_candidate_job_registry(profile="safe", registry=registry)
    }["local_registered__audio_separator_bs_roformer_viperx_1297_vendor_config_vocals__both__minus_direct"]

    assert default_config.command_template == overridden_config.command_template
    assert default_config.source_config_hash == overridden_config.source_config_hash
