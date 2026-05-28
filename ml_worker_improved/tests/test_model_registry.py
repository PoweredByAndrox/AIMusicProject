from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from worker.model_registry import ModelRegistry, ModelRegistryEntry, default_model_registry


def test_importing_model_registry_does_not_eager_load_queue_worker_infra() -> None:
    sys.modules.pop("worker.queue_worker", None)

    assert "worker.queue_worker" not in sys.modules


def test_worker_lazy_exports_preserve_queue_worker_public_import(monkeypatch) -> None:
    sys.modules.pop("worker.queue_worker", None)
    redis_module = types.ModuleType("redis")
    redis_module.Redis = type("Redis", (), {})
    psycopg2_module = types.ModuleType("psycopg2")
    psycopg2_module.extensions = types.SimpleNamespace(connection=object)
    psycopg2_extras_module = types.ModuleType("psycopg2.extras")
    psycopg2_extras_module.RealDictCursor = type("RealDictCursor", (), {})
    monkeypatch.setitem(sys.modules, "redis", redis_module)
    monkeypatch.setitem(sys.modules, "psycopg2", psycopg2_module)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", psycopg2_extras_module)

    worker_package = importlib.import_module("worker")

    assert worker_package.__all__ == ["QueueWorker", "AudioAnalyzer"]
    assert "worker.queue_worker" not in sys.modules
    assert worker_package.QueueWorker.__name__ == "QueueWorker"
    assert "worker.queue_worker" in sys.modules
    sys.modules.pop("worker.queue_worker", None)


def test_default_registry_contains_known_production_model_families() -> None:
    registry = default_model_registry()
    model_ids = {entry.model_id for entry in registry.entries}

    assert "htdemucs_ft" in model_ids
    assert "htdemucs_6s" in model_ids
    assert "htdemucs_6s_aggressive_melody_first" in model_ids
    assert "hybrid_htdemucs_ft_bs_roformer_6s" in model_ids
    assert "hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first" in model_ids
    assert "scnet_xl_ihf_musdb18" in model_ids
    assert "mdx23c_vocals" in model_ids
    assert "melband_roformer_vocals" in model_ids
    assert "audio_separator_melband_roformer_viperx_vocals" in model_ids
    assert "becruily_karaoke_vocals" in model_ids
    assert "bs_mamba2_vocals" in model_ids


def test_aggressive_melody_first_registry_routes_are_full_track_drum_candidates() -> None:
    registry = default_model_registry()
    entry = registry.get("htdemucs_6s_aggressive_melody_first")

    assert "/app/tools/export_hq_stems_full.py" in entry.command_template
    assert "--aggressive-melody-first" in entry.command_template
    assert "--model-name" in entry.command_template
    assert "htdemucs_6s" in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert "drums" in entry.target_stems


def test_tonal_harmonic_melody_registry_route_is_no_reference_candidate() -> None:
    registry = default_model_registry()
    entry = registry.get("tonal_harmonic_melody_v1")

    assert entry.family == "dsp"
    assert entry.target_stems == ("melody",)
    assert entry.output_stems == ("melody",)
    assert "/app/tools/build_tonal_harmonic_melody_candidate.py" in entry.command_template
    assert "--listen-dir={listen_dir}" in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert "drum_tonal_leakage" in entry.issue_labels_addressed
    assert "non_target_leakage" in entry.issue_labels_addressed
    assert entry.cost_class == "medium"


def test_model_stack_melody_registry_route_is_no_reference_candidate() -> None:
    registry = default_model_registry()
    entry = registry.get("model_stack_melody_v1")

    assert entry.family == "hybrid"
    assert entry.target_stems == ("melody",)
    assert entry.output_stems == ("melody",)
    assert "/app/tools/build_model_stack_melody_candidate.py" in entry.command_template
    assert "--listen-dir={listen_dir}" in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert "vocal_residue_in_melody" in entry.issue_labels_addressed
    assert "micro_reference_stem_leakage" in entry.issue_labels_addressed
    assert entry.cost_class == "high"


def test_joint_vocal_melody_registry_route_is_no_reference_candidate() -> None:
    registry = default_model_registry()
    entry = registry.get("joint_vocal_melody_v1")

    assert entry.family == "hybrid"
    assert entry.target_stems == ("vocals", "melody")
    assert entry.output_stems == ("vocals", "melody")
    assert "/app/tools/build_joint_vocal_melody_candidate.py" in entry.command_template
    assert "--listen-dir={listen_dir}" in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert "vocal_residue_in_melody" in entry.issue_labels_addressed
    assert "melody_leakage_in_vocals" in entry.issue_labels_addressed
    assert entry.cost_class == "high"


def test_joint_vocal_melody_v2_registry_route_is_no_reference_candidate() -> None:
    registry = default_model_registry()
    entry = registry.get("joint_vocal_melody_v2")

    assert entry.family == "hybrid"
    assert entry.target_stems == ("vocals", "melody")
    assert entry.output_stems == ("vocals", "melody")
    assert "/app/tools/build_joint_vocal_melody_candidate_v2.py" in entry.command_template
    assert "--listen-dir={listen_dir}" in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert "vocal_residue_in_melody" in entry.issue_labels_addressed
    assert "melody_leakage_in_vocals" in entry.issue_labels_addressed
    assert "non_target_leakage" in entry.issue_labels_addressed
    assert "exact_reference_threshold_failed" in entry.issue_labels_addressed
    assert "sample_residual_above_threshold" in entry.issue_labels_addressed
    assert entry.cost_class == "high"


def test_joint_vocal_melody_v3_registry_route_is_no_reference_candidate() -> None:
    registry = default_model_registry()
    entry = registry.get("joint_vocal_melody_v3")

    assert entry.family == "hybrid"
    assert entry.target_stems == ("vocals", "melody")
    assert entry.output_stems == ("vocals", "melody")
    assert "/app/tools/build_joint_vocal_melody_candidate_v3.py" in entry.command_template
    assert "--listen-dir={listen_dir}" in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert "exact_reference_threshold_failed" in entry.issue_labels_addressed
    assert "sample_residual_above_threshold" in entry.issue_labels_addressed
    assert "vocal_residue_in_melody" in entry.issue_labels_addressed
    assert "melody_leakage_in_vocals" in entry.issue_labels_addressed
    assert "cross_stem_leakage" in entry.issue_labels_addressed
    assert entry.cost_class == "high"


def test_drums_plus_percussive_fx_aux_registry_route_is_no_reference_candidate() -> None:
    registry = default_model_registry()
    entry = registry.get("htdemucs_6s_drums_plus_percussive_fx_aux")
    ft_entry = registry.get("htdemucs_ft_drums_plus_percussive_fx_aux")

    assert entry.target_stems == ("drums_plus_percussive_fx_aux",)
    assert entry.output_stems == ("drums_plus_percussive_fx_aux",)
    assert "/app/tools/export_drums_plus_percussive_fx_aux.py" in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert "harmonic_leakage_in_drums" in entry.issue_labels_addressed
    assert "bar_spanning_drop_melody_leakage" in entry.issue_labels_addressed
    assert "reference_stem_leakage" in entry.issue_labels_addressed
    assert entry.cost_class == "high"
    assert ft_entry.target_stems == ("drums_plus_percussive_fx_aux",)
    assert "htdemucs_ft" in ft_entry.command_template


def test_scnet_xl_registry_uses_full_track_multistem_exporter() -> None:
    registry = default_model_registry()
    entry = registry.get("scnet_xl_ihf_musdb18")

    assert "/app/tools/export_scnet_full.py" in entry.command_template
    assert "--listen-dir={listen_dir}" in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert entry.target_stems == ("drums", "bass", "other", "vocals")
    assert entry.output_stems == ("drums", "bass", "other", "vocals")
    assert "/app/vendor/music_source_separation_training" in entry.required_paths
    assert "/app/vendor/scnet_xl_ihf/config_musdb18_scnet_xl_more_wide_v5.yaml" in entry.required_paths
    assert "/app/vendor/scnet_xl_ihf/model_scnet_ep_36_sdr_10.0891.ckpt" in entry.required_paths
    assert "drum_tonal_leakage" in entry.issue_labels_addressed
    assert "reference_stem_leakage" in entry.issue_labels_addressed
    assert "harmonic_leakage_in_drums" in entry.issue_labels_addressed
    assert "micro_reference_stem_leakage" in entry.issue_labels_addressed
    assert "micro_transient_mismatch" in entry.issue_labels_addressed
    assert "sample_residual_above_threshold" in entry.issue_labels_addressed

    command_without_listen = entry.render_command(
        source_audio_path="/app/input/source.flac",
        output_dir="/app/output/scnet",
        listen_dir=None,
    )
    command_with_listen = entry.render_command(
        source_audio_path="/app/input/source.flac",
        output_dir="/app/output/scnet",
        listen_dir="/app/listen/scnet",
    )

    assert "--listen-dir=" not in " ".join(command_without_listen)
    assert "--listen-dir=/app/listen/scnet" in command_with_listen


def test_hybrid_aggressive_melody_first_registry_declares_bs_runtime_paths() -> None:
    registry = default_model_registry()
    entry = registry.get("hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first")

    assert "--aggressive-melody-first" in entry.command_template
    assert "hybrid_htdemucs_ft_bs_roformer_6s" in entry.command_template
    assert "/app/vendor/music_source_separation_training" in entry.required_paths
    assert "/app/vendor/bs_roformer_official/config_bs_roformer_384_8_2_485100.yaml" in entry.required_paths
    assert "/app/vendor/bs_roformer_official/model_bs_roformer_ep_17_sdr_9.6568.ckpt" in entry.required_paths
    assert "drums" in entry.target_stems


def test_melband_registry_uses_full_track_benchmark_exporter() -> None:
    registry = default_model_registry()
    entry = registry.get("melband_roformer_vocals")

    assert "/app/tools/export_melband_vocal_full.py" in entry.command_template
    assert "/app/tools/export_melband_vocal_preview.py" not in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert entry.target_stems == ("vocals", "instrumental")


def test_report_metadata_marks_vocal_only_models_as_not_comparable_to_four_stem_scores() -> None:
    registry = default_model_registry()
    melband = registry.get("melband_roformer_vocals")
    scnet = registry.get("scnet_xl_ihf_musdb18")

    assert melband.task_scope == "vocal_instrumental"
    assert melband.benchmark_score_context["metric_family"] == "vocal_only_sdr"
    assert melband.benchmark_score_context["directly_comparable_with"] == "vocal_only"
    assert melband.license_status == "checkpoint_license_unverified"
    assert "not directly comparable" in " ".join(melband.comparability_notes).lower()

    assert scnet.task_scope == "four_stem"
    assert scnet.benchmark_score_context["metric_family"] == "four_stem_average_sdr"
    assert scnet.benchmark_score_context["directly_comparable_with"] == "four_stem_average"


def test_registry_manifest_exposes_report_metadata_for_auditability() -> None:
    registry = default_model_registry()
    manifest = registry.to_manifest()
    by_id = {entry["model_id"]: entry for entry in manifest["entries"]}

    melband = by_id["melband_roformer_vocals"]
    assert melband["task_scope"] == "vocal_instrumental"
    assert melband["license_status"] == "checkpoint_license_unverified"
    assert melband["code_license"] == "wrapper_mit"
    assert melband["weights_license"] == "checkpoint_specific"
    assert melband["commercial_use_status"] == "requires_checkpoint_rights_review"
    assert melband["canonical_name_status"] == "paper_family_with_project_checkpoint"
    assert melband["benchmark_score_context"]["metric_family"] == "vocal_only_sdr"
    assert melband["comparability_notes"]
    assert melband["report_alignment_notes"]


def test_becruily_karaoke_registry_uses_melband_full_exporter_with_truthful_manifest_metadata() -> None:
    registry = default_model_registry()
    entry = registry.get("becruily_karaoke_vocals")

    assert "/app/tools/export_melband_vocal_full.py" in entry.command_template
    assert "--model-path" in entry.command_template
    assert "/app/vendor/becruily_karaoke/mel_band_roformer_karaoke_becruily.ckpt" in entry.command_template
    assert "--config-path" in entry.command_template
    assert "/app/vendor/becruily_karaoke/config_karaoke_becruily.yaml" in entry.command_template
    assert "--manifest-model-id" in entry.command_template
    assert "becruily_karaoke_vocals" in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert entry.target_stems == ("vocals", "instrumental")


def test_mdx23c_registry_uses_full_track_benchmark_exporter() -> None:
    registry = default_model_registry()
    entry = registry.get("mdx23c_vocals")

    assert "/app/tools/export_mdx23c_vocal_full.py" in entry.command_template
    assert "/app/tools/export_mdx23c_vocal_preview.py" not in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert entry.target_stems == ("vocals", "instrumental")


def test_bs_mamba2_registry_uses_full_track_benchmark_exporter() -> None:
    registry = default_model_registry()
    entry = registry.get("bs_mamba2_vocals")

    assert "/app/tools/export_bs_mamba2_vocal_full.py" in entry.command_template
    assert "/app/tools/export_bs_mamba2_vocal_preview.py" not in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert entry.target_stems == ("vocals", "instrumental")


def test_audio_separator_bs_roformer_registry_uses_uvr_full_track_exporter() -> None:
    registry = default_model_registry()
    entry = registry.get("audio_separator_bs_roformer_vocals")

    assert "/app/tools/export_audio_separator_vocal_full.py" in entry.command_template
    assert "--model-filename" in entry.command_template
    assert "model_bs_roformer_ep_317_sdr_12.9755.ckpt" in entry.command_template
    assert "--model-file-dir" in entry.command_template
    assert "/app/models/audio_separator" in entry.command_template
    assert "--audio-separator-python" in entry.command_template
    assert "/opt/audio-separator-venv/bin/python" in entry.command_template
    assert "--require-cuda-provider" in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert entry.target_stems == ("vocals", "instrumental")
    assert entry.required_paths == (
        "/opt/audio-separator-venv/bin/python",
        "/app/models/audio_separator/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "/app/models/audio_separator/model_bs_roformer_ep_317_sdr_12.9755.yaml",
    )
    assert entry.required_python_modules == ()


def test_audio_separator_melband_viperx_registry_uses_uvr_full_track_exporter() -> None:
    registry = default_model_registry()
    entry = registry.get("audio_separator_melband_roformer_viperx_vocals")

    assert "/app/tools/export_audio_separator_vocal_full.py" in entry.command_template
    assert "--model-filename" in entry.command_template
    assert "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt" in entry.command_template
    assert "--model-file-dir" in entry.command_template
    assert "/app/vendor/melband_roformer_viperx_11436" in entry.command_template
    assert "--audio-separator-python" in entry.command_template
    assert "/opt/audio-separator-venv/bin/python" in entry.command_template
    assert "--manifest-model-id" in entry.command_template
    assert "audio_separator_melband_roformer_viperx_vocals" in entry.command_template
    assert "--require-cuda-provider" in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert entry.target_stems == ("vocals", "instrumental")
    assert entry.output_stems == ("vocals", "instrumental")
    assert entry.required_paths == (
        "/opt/audio-separator-venv/bin/python",
        "/app/vendor/melband_roformer_viperx_11436/model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        "/app/vendor/melband_roformer_viperx_11436/model_mel_band_roformer_ep_3005_sdr_11.4360.yaml",
    )
    assert entry.required_python_modules == ()
    assert "drop_melody_leakage" in entry.issue_labels_addressed
    assert "high_noise_or_robotic_artifact" in entry.issue_labels_addressed


def test_registry_filters_by_stem_labels_and_evaluated_aliases() -> None:
    registry = default_model_registry()

    suggestions = registry.suggest_next_candidates(
        stem_names=["vocals"],
        failed_labels=["reference_likeness_below_threshold", "vocal_dropout"],
        evaluated_labels=["mdx23c"],
        include_unavailable=True,
        limit=20,
    )

    suggested_ids = {suggestion.model_id for suggestion in suggestions}
    assert "mdx23c_vocals" not in suggested_ids
    assert "htdemucs_ft" in suggested_ids
    assert "melband_roformer_vocals" in suggested_ids
    assert all("vocals" in suggestion.target_stems for suggestion in suggestions)

    melband_suggestion = next(suggestion for suggestion in suggestions if suggestion.model_id == "melband_roformer_vocals")
    suggestion_manifest = melband_suggestion.to_dict()
    assert suggestion_manifest["task_scope"] == "vocal_instrumental"
    assert suggestion_manifest["license_status"] == "checkpoint_license_unverified"
    assert suggestion_manifest["code_license"] == "wrapper_mit"
    assert suggestion_manifest["weights_license"] == "checkpoint_specific"
    assert suggestion_manifest["commercial_use_status"] == "requires_checkpoint_rights_review"
    assert suggestion_manifest["canonical_name_status"] == "paper_family_with_project_checkpoint"
    assert suggestion_manifest["benchmark_score_context"]["metric_family"] == "vocal_only_sdr"
    assert suggestion_manifest["comparability_notes"]
    assert suggestion_manifest["report_alignment_notes"]


def test_registry_suggests_drums_plus_percussive_fx_aux_route() -> None:
    registry = default_model_registry()

    suggestions = registry.suggest_next_candidates(
        stem_names=["drums_plus_percussive_fx_aux"],
        failed_labels=["bar_spanning_drop_melody_leakage"],
        evaluated_labels=[],
        include_unavailable=True,
        limit=20,
    )

    suggested_ids = {suggestion.model_id for suggestion in suggestions}
    assert "htdemucs_6s_drums_plus_percussive_fx_aux" in suggested_ids
    assert "htdemucs_ft_drums_plus_percussive_fx_aux" in suggested_ids
    assert "hybrid_htdemucs_ft_bs_roformer_6s_drums_plus_percussive_fx_aux" in suggested_ids
    assert all(
        "drums_plus_percussive_fx_aux" in suggestion.target_stems
        for suggestion in suggestions
    )


def test_querysep_strings_orchestra_registry_route_is_no_reference_candidate() -> None:
    registry = default_model_registry()
    entry = registry.get("querysep_banquet_strings_orchestra_v1")

    assert entry.family == "querysep"
    assert entry.target_stems == ("strings_orchestra",)
    assert entry.output_stems == ("strings_orchestra",)
    assert "/app/tools/build_querysep_strings_orchestra_candidate.py" in entry.command_template
    assert "--listen-dir={listen_dir}" in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert "possible_other_proxy_leakage" in entry.issue_labels_addressed
    assert "reference_likeness_below_threshold" in entry.issue_labels_addressed
    assert entry.cost_class == "high"


def test_model_stack_strings_orchestra_registry_route_is_no_reference_candidate() -> None:
    registry = default_model_registry()
    entry = registry.get("model_stack_strings_orchestra_v1")

    assert entry.family == "hybrid"
    assert entry.target_stems == ("strings_orchestra",)
    assert entry.output_stems == ("strings_orchestra",)
    assert "/app/tools/build_model_stack_strings_orchestra_candidate.py" in entry.command_template
    assert "--listen-dir={listen_dir}" in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert "possible_other_proxy_leakage" in entry.issue_labels_addressed
    assert "long_tail_presence_too_weak" in entry.issue_labels_addressed
    assert "reference_likeness_below_threshold" in entry.issue_labels_addressed
    assert entry.cost_class == "high"


def test_model_stack_strings_orchestra_hybrid_bs_registry_route_requires_stronger_base_assets() -> None:
    registry = default_model_registry()
    entry = registry.get("model_stack_strings_orchestra_hybrid_bs_v1")

    assert entry.family == "hybrid"
    assert entry.target_stems == ("strings_orchestra",)
    assert entry.output_stems == ("strings_orchestra",)
    assert "/app/tools/build_model_stack_strings_orchestra_candidate.py" in entry.command_template
    assert "--model-name" in entry.command_template
    assert "hybrid_htdemucs_ft_bs_roformer_6s" in entry.command_template
    assert "--route-name" in entry.command_template
    assert "model_stack_strings_orchestra_hybrid_bs_v1" in entry.command_template
    assert "--listen-dir={listen_dir}" in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert "/app/vendor/music_source_separation_training" in entry.required_paths
    assert "/app/vendor/bs_roformer_official/config_bs_roformer_384_8_2_485100.yaml" in entry.required_paths
    assert "/app/vendor/bs_roformer_official/model_bs_roformer_ep_17_sdr_9.6568.ckpt" in entry.required_paths
    assert "spectral_timbre_mismatch" in entry.issue_labels_addressed
    assert "reference_likeness_below_threshold" in entry.issue_labels_addressed
    assert entry.cost_class == "high"


def test_registry_suggests_querysep_strings_orchestra_route_for_long_tail_failures() -> None:
    registry = default_model_registry()

    suggestions = registry.suggest_next_candidates(
        stem_names=["strings_orchestra"],
        failed_labels=[
            "reference_likeness_below_threshold",
            "possible_other_proxy_leakage",
            "long_tail_presence_too_weak",
        ],
        evaluated_labels=[],
        include_unavailable=True,
        limit=20,
    )

    suggested_ids = {suggestion.model_id for suggestion in suggestions}
    assert "querysep_banquet_strings_orchestra_v1" in suggested_ids
    assert all("strings_orchestra" in suggestion.target_stems for suggestion in suggestions)


def test_registry_suggests_model_stack_strings_orchestra_after_querysep_failure() -> None:
    registry = default_model_registry()

    suggestions = registry.suggest_next_candidates(
        stem_names=["strings_orchestra"],
        failed_labels=[
            "reference_likeness_below_threshold",
            "long_tail_presence_too_weak",
            "possible_other_proxy_leakage",
        ],
        evaluated_labels=["querysep_banquet_strings_orchestra_v1"],
        include_unavailable=True,
        limit=20,
    )

    suggested_ids = {suggestion.model_id for suggestion in suggestions}
    assert "model_stack_strings_orchestra_v1" in suggested_ids
    assert "querysep_banquet_strings_orchestra_v1" not in suggested_ids
    assert all("strings_orchestra" in suggestion.target_stems for suggestion in suggestions)


def test_registry_suggests_hybrid_bs_strings_after_base_stack_failure() -> None:
    registry = default_model_registry()

    suggestions = registry.suggest_next_candidates(
        stem_names=["strings_orchestra"],
        failed_labels=[
            "reference_likeness_below_threshold",
            "spectral_timbre_mismatch",
            "sample_residual_above_threshold",
        ],
        evaluated_labels=[
            "querysep_banquet_strings_orchestra_v1",
            "model_stack_strings_orchestra_v1",
        ],
        include_unavailable=True,
        limit=20,
    )

    suggested_ids = {suggestion.model_id for suggestion in suggestions}
    assert "model_stack_strings_orchestra_hybrid_bs_v1" in suggested_ids
    assert "model_stack_strings_orchestra_v1" not in suggested_ids
    assert all("strings_orchestra" in suggestion.target_stems for suggestion in suggestions)


def test_registry_suggests_tonal_harmonic_melody_route_for_melody_drop_labels() -> None:
    registry = default_model_registry()

    suggestions = registry.suggest_next_candidates(
        stem_names=["melody"],
        failed_labels=["bar_spanning_drop_melody_leakage", "non_target_leakage"],
        evaluated_labels=[],
        include_unavailable=True,
        limit=20,
    )

    suggested_ids = {suggestion.model_id for suggestion in suggestions}
    assert "tonal_harmonic_melody_v1" in suggested_ids
    assert all("melody" in suggestion.target_stems for suggestion in suggestions)


def test_registry_suggests_model_stack_melody_route_for_vocal_residue_labels() -> None:
    registry = default_model_registry()

    suggestions = registry.suggest_next_candidates(
        stem_names=["melody"],
        failed_labels=[
            "reference_likeness_below_threshold",
            "vocal_residue_in_melody",
            "micro_reference_stem_leakage",
        ],
        evaluated_labels=["tonal_harmonic_melody_v1"],
        include_unavailable=True,
        limit=20,
    )

    suggested_ids = {suggestion.model_id for suggestion in suggestions}
    assert "model_stack_melody_v1" in suggested_ids
    assert "tonal_harmonic_melody_v1" not in suggested_ids


def test_registry_suggests_joint_vocal_melody_route_for_cross_leakage_labels() -> None:
    registry = default_model_registry()

    suggestions = registry.suggest_next_candidates(
        stem_names=["vocals", "melody"],
        failed_labels=[
            "vocal_residue_in_melody",
            "melody_leakage_in_vocals",
            "cross_stem_leakage",
        ],
        evaluated_labels=["model_stack_melody_v1"],
        include_unavailable=True,
        limit=20,
    )

    suggested_ids = {suggestion.model_id for suggestion in suggestions}
    assert "joint_vocal_melody_v1" in suggested_ids
    assert "model_stack_melody_v1" not in suggested_ids


def test_registry_suggests_joint_vocal_melody_v2_after_v1_exact_audit_failure() -> None:
    registry = default_model_registry()

    suggestions = registry.suggest_next_candidates(
        stem_names=["vocals", "melody"],
        failed_labels=[
            "exact_reference_threshold_failed",
            "sample_residual_above_threshold",
            "vocal_residue_in_melody",
            "melody_leakage_in_vocals",
        ],
        evaluated_labels=["model_stack_melody_v1", "joint_vocal_melody_v1"],
        include_unavailable=True,
        limit=20,
    )

    suggested_ids = {suggestion.model_id for suggestion in suggestions}
    assert "joint_vocal_melody_v2" in suggested_ids
    assert "joint_vocal_melody_v1" not in suggested_ids
    assert "model_stack_melody_v1" not in suggested_ids


def test_registry_suggests_joint_vocal_melody_v3_after_v2_exact_audit_failure() -> None:
    registry = default_model_registry()

    suggestions = registry.suggest_next_candidates(
        stem_names=["vocals", "melody"],
        failed_labels=[
            "exact_reference_threshold_failed",
            "sample_residual_above_threshold",
            "vocal_residue_in_melody",
            "melody_leakage_in_vocals",
        ],
        evaluated_labels=["model_stack_melody_v1", "joint_vocal_melody_v1", "joint_vocal_melody_v2"],
        include_unavailable=True,
        limit=20,
    )

    suggested_ids = {suggestion.model_id for suggestion in suggestions}
    assert "joint_vocal_melody_v3" in suggested_ids
    assert "joint_vocal_melody_v2" not in suggested_ids
    assert "joint_vocal_melody_v1" not in suggested_ids


def test_hybrid_drums_plus_percussive_fx_aux_registry_route_uses_full_mix_exporter() -> None:
    registry = default_model_registry()
    entry = registry.get("hybrid_htdemucs_ft_bs_roformer_6s_drums_plus_percussive_fx_aux")

    assert entry.family == "hybrid"
    assert entry.target_stems == ("drums_plus_percussive_fx_aux",)
    assert entry.output_stems == ("drums_plus_percussive_fx_aux",)
    assert "/app/tools/export_drums_plus_percussive_fx_aux.py" in entry.command_template
    assert "--model-name" in entry.command_template
    assert "hybrid_htdemucs_ft_bs_roformer_6s" in entry.command_template
    assert "--listen-dir={listen_dir}" in entry.command_template
    assert entry.expected_manifest == "{output_dir}/stem_manifest.json"
    assert "/app/vendor/music_source_separation_training" in entry.required_paths
    assert "/app/vendor/bs_roformer_official/config_bs_roformer_384_8_2_485100.yaml" in entry.required_paths
    assert "/app/vendor/bs_roformer_official/model_bs_roformer_ep_17_sdr_9.6568.ckpt" in entry.required_paths
    assert "bar_spanning_drop_melody_leakage" in entry.issue_labels_addressed
    assert "micro_transient_mismatch" in entry.issue_labels_addressed


def test_registry_availability_reports_missing_required_paths_without_fallback(tmp_path: Path) -> None:
    registry = ModelRegistry(
        runtime_root=tmp_path,
        entries=(
            ModelRegistryEntry(
                model_id="demo",
                display_name="Demo",
                family="test",
                target_stems=("vocals",),
                output_stems=("vocals",),
                command_template=("python", "demo.py"),
                expected_manifest="{output_dir}/stem_manifest.json",
                required_paths=("models/demo.ckpt",),
            ),
        ),
    )

    availability = registry.availability(registry.get("demo"))
    assert availability.status == "unavailable"
    assert availability.missing_paths == ("models/demo.ckpt",)

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "demo.ckpt").write_text("stub", encoding="utf-8")

    availability = registry.availability(registry.get("demo"))
    assert availability.status == "available"
    assert availability.missing_paths == ()


def test_registry_availability_reports_missing_required_python_modules(tmp_path: Path) -> None:
    registry = ModelRegistry(
        runtime_root=tmp_path,
        entries=(
            ModelRegistryEntry(
                model_id="demo_python_module",
                display_name="Demo Python module",
                family="test",
                target_stems=("vocals",),
                output_stems=("vocals",),
                command_template=("python", "demo.py"),
                expected_manifest="{output_dir}/stem_manifest.json",
                required_python_modules=("definitely_missing_audio_dependency_for_test",),
            ),
        ),
    )

    availability = registry.availability(registry.get("demo_python_module"))
    assert availability.status == "unavailable"
    assert availability.missing_paths == ()
    assert availability.missing_python_modules == ("definitely_missing_audio_dependency_for_test",)


def test_bs_mamba2_registry_declares_mamba_runtime_dependencies() -> None:
    registry = default_model_registry()
    entry = registry.get("bs_mamba2_vocals")

    assert entry.required_python_modules == ("mamba_ssm", "causal_conv1d")


def test_registry_renders_command_without_track_specific_state() -> None:
    registry = default_model_registry()
    entry = registry.get("htdemucs_6s")

    command = entry.render_command(
        source_audio_path="/app/input/source.flac",
        output_dir="/app/output/htdemucs_6s",
        device="cuda",
    )

    assert "/app/tools/export_hq_stems_full.py" in command
    assert "htdemucs_6s" in command
    assert "/app/input/source.flac" in command
    assert "/app/output/htdemucs_6s" in command
