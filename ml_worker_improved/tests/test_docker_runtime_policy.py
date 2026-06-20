from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

REQUIRED_DOCKER_ONLY_ENTRYPOINTS = (
    SCRIPTS_ROOT / "run_ground_truth_quality_gate.py",
    SCRIPTS_ROOT / "run_nine_thou_quality_gate.py",
    SCRIPTS_ROOT / "build_instrumental_from_stem_manifest.py",
    SCRIPTS_ROOT / "compare_instrumental_to_reference_pair.py",
    SCRIPTS_ROOT / "probe_golden_reference_pair.py",
    SCRIPTS_ROOT / "virtualdj_baseline_manifest.py",
    TOOLS_ROOT / "adaptive_minus_repair.py",
    TOOLS_ROOT / "adaptive_vocal_isolation.py",
    TOOLS_ROOT / "apply_bs_demucs_focuspatch.py",
    TOOLS_ROOT / "apply_drum_transient_restore.py",
    TOOLS_ROOT / "apply_flexible_minus_repair.py",
    TOOLS_ROOT / "apply_hybrid_minus_guided_cleanup.py",
    TOOLS_ROOT / "apply_judge_guided_minus_vocal_leakage_repair.py",
    TOOLS_ROOT / "apply_manual_window_demucs_patch.py",
    TOOLS_ROOT / "apply_minus_vocal_residue_cleanup.py",
    TOOLS_ROOT / "apply_vocal_harshness_guard.py",
    TOOLS_ROOT / "apply_windowed_drum_body_restore.py",
    TOOLS_ROOT / "apply_windowed_hybrid_minus_cleanup.py",
    TOOLS_ROOT / "apply_windowed_vocal_intelligibility_restore.py",
    TOOLS_ROOT / "batch_export_hq_stem_previews.py",
    TOOLS_ROOT / "batch_export_hq_vocal_previews.py",
    TOOLS_ROOT / "build_aris_idea_funnel.py",
    TOOLS_ROOT / "build_metadata_regression.py",
    TOOLS_ROOT / "build_flstudio_ground_truth_bundle.py",
    TOOLS_ROOT / "build_complex_oracle_mask_teacher.py",
    TOOLS_ROOT / "build_fullmix_residual_percussive_aux_candidate.py",
    TOOLS_ROOT / "build_percussive_fx_aux_candidate.py",
    TOOLS_ROOT / "build_percussive_fx_aux_profile_candidate.py",
    TOOLS_ROOT / "build_querysep_strings_orchestra_candidate.py",
    TOOLS_ROOT / "build_reco_regression.py",
    TOOLS_ROOT / "build_no_reference_corpus_route_priors.py",
    TOOLS_ROOT / "build_release_delivery_manifest.py",
    TOOLS_ROOT / "build_reference_guided_band_repair.py",
    TOOLS_ROOT / "build_reference_aux_overlay_candidate.py",
    TOOLS_ROOT / "build_reference_guided_stem_residual_repair.py",
    TOOLS_ROOT / "build_reference_guided_vocal_ensemble.py",
    TOOLS_ROOT / "build_reference_teacher_stem_pack.py",
    TOOLS_ROOT / "build_stem_delta_profile_repair.py",
    TOOLS_ROOT / "build_teacher_profile_bank.py",
    TOOLS_ROOT / "build_testing_stuff_music_corpus_job_matrix.py",
    TOOLS_ROOT / "build_testing_stuff_music_corpus_plan.py",
    TOOLS_ROOT / "build_vocal_guided_minus.py",
    TOOLS_ROOT / "build_vocal_model_zoo_candidates.py",
    TOOLS_ROOT / "calibrate_segment_router_confidence.py",
    TOOLS_ROOT / "detect_vocal_window.py",
    TOOLS_ROOT / "drum_audit_guided_residual_debleed.py",
    TOOLS_ROOT / "drum_bus_projection_body_gain.py",
    TOOLS_ROOT / "drum_micro_leakage_debleed.py",
    TOOLS_ROOT / "export_bs_mamba2_minus_preview.py",
    TOOLS_ROOT / "export_bs_mamba2_vocal_full.py",
    TOOLS_ROOT / "export_bs_mamba2_vocal_preview.py",
    TOOLS_ROOT / "export_demucs_minus_preview.py",
    TOOLS_ROOT / "export_drums_plus_percussive_fx_aux.py",
    TOOLS_ROOT / "export_dereverb_vocal_preview.py",
    TOOLS_ROOT / "export_hq_stem_preview.py",
    TOOLS_ROOT / "export_hq_stem_rewrite_full.py",
    TOOLS_ROOT / "export_hq_stems_full.py",
    TOOLS_ROOT / "export_hq_vocal_preview.py",
    TOOLS_ROOT / "export_instrumental_resurrection_minus.py",
    TOOLS_ROOT / "export_mdx23c_vocal_full.py",
    TOOLS_ROOT / "export_mdx23c_vocal_preview.py",
    TOOLS_ROOT / "export_melband_vocal_full.py",
    TOOLS_ROOT / "export_melband_vocal_preview.py",
    TOOLS_ROOT / "export_scnet_full.py",
    TOOLS_ROOT / "export_vocal_window_fusion_preview.py",
    TOOLS_ROOT / "drum_tonal_leakage_repair.py",
    TOOLS_ROOT / "list_model_registry.py",
    TOOLS_ROOT / "localize_reference_quality_failures.py",
    TOOLS_ROOT / "manage_artifact_rag.py",
    TOOLS_ROOT / "materialize_aris_rag_wave_metadata.py",
    TOOLS_ROOT / "oracle_stem_upper_bound.py",
    TOOLS_ROOT / "query_artifact_rag.py",
    TOOLS_ROOT / "reference_guided_drum_candidate_router.py",
    TOOLS_ROOT / "reference_guided_drum_reconstructor.py",
    TOOLS_ROOT / "repair_hq_stem_exports.py",
    TOOLS_ROOT / "run_adaptive_vocal_contract_batch.py",
    TOOLS_ROOT / "run_metadata_regression.py",
    TOOLS_ROOT / "route_drum_model_zoo_segments.py",
    TOOLS_ROOT / "run_model_zoo_benchmark.py",
    TOOLS_ROOT / "run_aris_refinement_matrix.py",
    TOOLS_ROOT / "run_aris_paired_gpu_route.py",
    TOOLS_ROOT / "build_aris_refinement_candidate_pool.py",
    TOOLS_ROOT / "build_aris_top1_top2_plan.py",
    TOOLS_ROOT / "run_posthoc_release_evaluator.py",
    TOOLS_ROOT / "run_reco_regression.py",
    TOOLS_ROOT / "run_registered_model_candidate.py",
    TOOLS_ROOT / "run_stem_quality_benchmark.py",
    TOOLS_ROOT / "run_testing_stuff_music_corpus_job_matrix.py",
    TOOLS_ROOT / "stem_quality_control.py",
    TOOLS_ROOT / "teacher_dictionary_drum_separator.py",
    TOOLS_ROOT / "validate_segment_router_confidence_holdout.py",
    TOOLS_ROOT / "waveform_forensic_compare.py",
)


def _load_docker_runtime_module():
    module_path = TOOLS_ROOT / "docker_runtime.py"
    spec = importlib.util.spec_from_file_location("docker_runtime", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_main_function(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("missing main()")


def _first_runtime_statement(function: ast.FunctionDef) -> ast.stmt:
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if not body:
        raise AssertionError("main() has no body")
    return body[0]


def _is_require_docker_runtime_call(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    call = statement.value.func
    return isinstance(call, ast.Name) and call.id == "require_docker_runtime"


def _defines_local_docker_runtime_guard(tree: ast.Module) -> bool:
    return any(isinstance(node, ast.FunctionDef) and node.name == "require_docker_runtime" for node in tree.body)


def _imports_shared_docker_runtime_guard(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module != "docker_runtime":
            continue
        if any(alias.name == "require_docker_runtime" for alias in node.names):
            return True
    return False


class TestDockerRuntimePolicy(unittest.TestCase):
    def test_shared_docker_runtime_guard_rejects_host_execution_with_compose_command(self):
        module = _load_docker_runtime_module()

        class MissingDockerRuntimePath:
            def __init__(self, value):
                self.value = value

            def exists(self):
                return False

        original_path = module.Path
        module.Path = MissingDockerRuntimePath
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "Docker-only.*docker compose run --rm --no-deps -T ml-worker",
            ):
                module.require_docker_runtime(
                    tool_name="test_tool.py",
                    entrypoint_path="/app/tools/test_tool.py",
                )
        finally:
            module.Path = original_path

    def test_shared_docker_runtime_guard_accepts_docker_marker(self):
        module = _load_docker_runtime_module()

        class PresentDockerRuntimePath:
            def __init__(self, value):
                self.value = value

            def exists(self):
                return self.value == "/.dockerenv"

        original_path = module.Path
        module.Path = PresentDockerRuntimePath
        try:
            module.require_docker_runtime(
                tool_name="test_tool.py",
                entrypoint_path="/app/tools/test_tool.py",
            )
        finally:
            module.Path = original_path

    def test_audio_entrypoints_start_main_with_docker_runtime_guard(self):
        for entrypoint in REQUIRED_DOCKER_ONLY_ENTRYPOINTS:
            with self.subTest(entrypoint=entrypoint.name):
                source = entrypoint.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(entrypoint))
                main_function = _find_main_function(tree)
                first_statement = _first_runtime_statement(main_function)
                self.assertTrue(
                    _is_require_docker_runtime_call(first_statement),
                    f"{entrypoint} must call require_docker_runtime() as the first main() statement",
                )

    def test_audio_entrypoints_use_shared_docker_runtime_guard(self):
        for entrypoint in REQUIRED_DOCKER_ONLY_ENTRYPOINTS:
            with self.subTest(entrypoint=entrypoint.name):
                source = entrypoint.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(entrypoint))
                self.assertFalse(
                    _defines_local_docker_runtime_guard(tree),
                    f"{entrypoint} must not define a local require_docker_runtime()",
                )
                self.assertTrue(
                    _imports_shared_docker_runtime_guard(tree),
                    f"{entrypoint} must import require_docker_runtime from docker_runtime",
                )
