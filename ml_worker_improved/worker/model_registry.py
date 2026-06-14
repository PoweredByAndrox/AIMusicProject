from __future__ import annotations

import json
import importlib.util
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_DOCKER_APP_ROOT = Path("/app")


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_id: str
    display_name: str
    family: str
    target_stems: tuple[str, ...]
    output_stems: tuple[str, ...]
    command_template: tuple[str, ...]
    expected_manifest: str
    required_paths: tuple[str, ...] = ()
    required_python_modules: tuple[str, ...] = ()
    issue_labels_addressed: tuple[str, ...] = ()
    runtime_tags: tuple[str, ...] = ("docker", "gpu")
    cost_class: str = "medium"
    notes: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    task_scope: str = "unspecified"
    canonical_name_status: str = "project_defined"
    benchmark_score_context: dict[str, Any] = field(default_factory=dict)
    code_license: str = "unknown"
    weights_license: str = "unknown"
    license_status: str = "unverified"
    commercial_use_status: str = "requires_review"
    comparability_notes: tuple[str, ...] = ()
    report_alignment_notes: tuple[str, ...] = ()

    def supports_any_stem(self, stem_names: Iterable[str]) -> bool:
        requested = {stem.lower() for stem in stem_names}
        supported = {stem.lower() for stem in self.target_stems}
        return bool(requested & supported)

    def addresses_any_label(self, labels: Iterable[str]) -> bool:
        requested = {label.lower() for label in labels}
        addressed = {label.lower() for label in self.issue_labels_addressed}
        return not requested or bool(requested & addressed)

    def render_command(
        self,
        *,
        source_audio_path: str,
        output_dir: str,
        listen_dir: str | None = None,
        device: str = "cuda",
    ) -> list[str]:
        replacements = {
            "source_audio_path": source_audio_path,
            "output_dir": output_dir,
            "listen_dir": listen_dir or "",
            "device": device,
        }
        rendered: list[str] = []
        for part in self.command_template:
            if "{listen_dir}" in part and not listen_dir:
                continue
            rendered.append(part.format(**replacements))
        return rendered


@dataclass(frozen=True)
class ModelAvailability:
    model_id: str
    status: str
    missing_paths: tuple[str, ...] = ()
    missing_python_modules: tuple[str, ...] = ()
    runtime_blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSuggestion:
    model_id: str
    display_name: str
    family: str
    target_stems: tuple[str, ...]
    output_stems: tuple[str, ...]
    availability: ModelAvailability
    reason: str
    command_template: tuple[str, ...]
    expected_manifest: str
    task_scope: str = "unspecified"
    benchmark_score_context: dict[str, Any] = field(default_factory=dict)
    license_status: str = "unverified"
    code_license: str = "unknown"
    weights_license: str = "unknown"
    commercial_use_status: str = "requires_review"
    canonical_name_status: str = "project_defined"
    comparability_notes: tuple[str, ...] = ()
    report_alignment_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "family": self.family,
            "target_stems": list(self.target_stems),
            "output_stems": list(self.output_stems),
                "availability": {
                    "status": self.availability.status,
                    "missing_paths": list(self.availability.missing_paths),
                    "missing_python_modules": list(self.availability.missing_python_modules),
                    "runtime_blockers": list(self.availability.runtime_blockers),
                },
            "reason": self.reason,
            "command_template": list(self.command_template),
            "expected_manifest": self.expected_manifest,
            "task_scope": self.task_scope,
            "benchmark_score_context": self.benchmark_score_context,
            "license_status": self.license_status,
            "code_license": self.code_license,
            "weights_license": self.weights_license,
            "commercial_use_status": self.commercial_use_status,
            "canonical_name_status": self.canonical_name_status,
            "comparability_notes": list(self.comparability_notes),
            "report_alignment_notes": list(self.report_alignment_notes),
        }


@dataclass(frozen=True)
class ModelRegistry:
    entries: tuple[ModelRegistryEntry, ...] = field(default_factory=tuple)
    runtime_root: Path = DEFAULT_DOCKER_APP_ROOT

    def get(self, model_id: str) -> ModelRegistryEntry:
        for entry in self.entries:
            if entry.model_id == model_id or model_id in entry.aliases:
                return entry
        raise KeyError(f"Unknown model id: {model_id}")

    def availability(self, entry: ModelRegistryEntry) -> ModelAvailability:
        missing_paths = tuple(
            required_path
            for required_path in entry.required_paths
            if not self._resolve_required_path(required_path).exists()
        )
        missing_python_modules = tuple(
            module_name
            for module_name in entry.required_python_modules
            if not self._python_module_is_available(module_name)
        )
        runtime_blockers = tuple(self._runtime_blockers(entry))
        return ModelAvailability(
            model_id=entry.model_id,
            status="available" if not missing_paths and not missing_python_modules and not runtime_blockers else "unavailable",
            missing_paths=missing_paths,
            missing_python_modules=missing_python_modules,
            runtime_blockers=runtime_blockers,
        )

    def supported_entries(self, stem_names: Sequence[str]) -> list[ModelRegistryEntry]:
        return [entry for entry in self.entries if entry.supports_any_stem(stem_names)]

    def suggest_next_candidates(
        self,
        *,
        stem_names: Sequence[str],
        failed_labels: Sequence[str],
        evaluated_labels: Sequence[str],
        include_unavailable: bool = True,
        limit: int = 5,
    ) -> list[ModelSuggestion]:
        evaluated = {label.lower() for label in evaluated_labels}
        suggestions: list[ModelSuggestion] = []
        for entry in self.supported_entries(stem_names):
            known_labels = {entry.model_id.lower(), *(alias.lower() for alias in entry.aliases)}
            if evaluated & known_labels:
                continue
            if not entry.addresses_any_label(failed_labels):
                continue
            availability = self.availability(entry)
            if availability.status != "available" and not include_unavailable:
                continue
            suggestions.append(
                ModelSuggestion(
                    model_id=entry.model_id,
                    display_name=entry.display_name,
                    family=entry.family,
                    target_stems=entry.target_stems,
                    output_stems=entry.output_stems,
                    availability=availability,
                    reason=build_suggestion_reason(entry, failed_labels),
                    command_template=entry.command_template,
                    expected_manifest=entry.expected_manifest,
                    task_scope=entry.task_scope,
                    benchmark_score_context=entry.benchmark_score_context,
                    license_status=entry.license_status,
                    code_license=entry.code_license,
                    weights_license=entry.weights_license,
                    commercial_use_status=entry.commercial_use_status,
                    canonical_name_status=entry.canonical_name_status,
                    comparability_notes=entry.comparability_notes,
                    report_alignment_notes=entry.report_alignment_notes,
                )
            )
        suggestions.sort(key=suggestion_sort_key)
        return suggestions[: max(0, int(limit))]

    def to_manifest(self) -> dict[str, Any]:
        def entry_to_manifest(entry: ModelRegistryEntry) -> dict[str, Any]:
            availability = self.availability(entry)
            return {
                "model_id": entry.model_id,
                "display_name": entry.display_name,
                "family": entry.family,
                "target_stems": list(entry.target_stems),
                "output_stems": list(entry.output_stems),
                "required_paths": list(entry.required_paths),
                "required_python_modules": list(entry.required_python_modules),
                "availability": {
                    "status": availability.status,
                    "missing_paths": list(availability.missing_paths),
                    "missing_python_modules": list(availability.missing_python_modules),
                    "runtime_blockers": list(availability.runtime_blockers),
                },
                "issue_labels_addressed": list(entry.issue_labels_addressed),
                "runtime_tags": list(entry.runtime_tags),
                "cost_class": entry.cost_class,
                "command_template": list(entry.command_template),
                "expected_manifest": entry.expected_manifest,
                "aliases": list(entry.aliases),
                "notes": list(entry.notes),
                "task_scope": entry.task_scope,
                "canonical_name_status": entry.canonical_name_status,
                "benchmark_score_context": entry.benchmark_score_context,
                "code_license": entry.code_license,
                "weights_license": entry.weights_license,
                "license_status": entry.license_status,
                "commercial_use_status": entry.commercial_use_status,
                "comparability_notes": list(entry.comparability_notes),
                "report_alignment_notes": list(entry.report_alignment_notes),
            }

        return {
            "schema_version": 1,
            "runtime_root": str(self.runtime_root),
            "entries": [entry_to_manifest(entry) for entry in self.entries],
        }

    def _resolve_required_path(self, required_path: str) -> Path:
        path = Path(required_path)
        if path.is_absolute():
            return path
        return self.runtime_root / path

    def _runtime_blockers(self, entry: ModelRegistryEntry) -> list[str]:
        command_template = list(entry.command_template)
        if "--require-cuda-provider" not in command_template:
            return []
        try:
            python_index = command_template.index("--audio-separator-python") + 1
            audio_separator_python = command_template[python_index]
        except (ValueError, IndexError):
            return ["audio_separator_python_arg_missing"]
        python_path = self._resolve_required_path(audio_separator_python)
        declared_runtime_paths = {
            str(self._resolve_required_path(required_path))
            for required_path in entry.required_paths
        }
        # Tests and diagnostic fixtures may strip required_paths to validate
        # command contracts without depending on the local audio-separator venv.
        # Production audio-separator entries declare this runtime path, so their
        # CUDA-provider probe remains fail-closed.
        if str(python_path) not in declared_runtime_paths:
            return []
        if not python_path.exists():
            return []
        providers, probe_error = _probe_onnxruntime_providers(str(python_path))
        if probe_error:
            return ["audio_separator_runtime_probe_failed"]
        if "CUDAExecutionProvider" not in providers:
            return ["cuda_execution_provider_missing"]
        return []

    @staticmethod
    def _python_module_is_available(module_name: str) -> bool:
        try:
            return importlib.util.find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False


@lru_cache(maxsize=16)
def _probe_onnxruntime_providers(audio_separator_python: str) -> tuple[tuple[str, ...], str]:
    runner_code = (
        "from __future__ import annotations\n"
        "import json\n"
        "try:\n"
        "    import onnxruntime as ort\n"
        "    print(json.dumps({'available_providers': list(ort.get_available_providers())}, ensure_ascii=False))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False))\n"
        "    raise\n"
    )
    completed = subprocess.run(
        [audio_separator_python, "-c", runner_code],
        capture_output=True,
        text=True,
        check=False,
    )
    last_json_line = next((line for line in reversed(completed.stdout.splitlines()) if line.strip().startswith("{")), "")
    if not last_json_line:
        return (), completed.stderr.strip() or f"probe_returncode={completed.returncode}"
    try:
        payload = json.loads(last_json_line)
    except json.JSONDecodeError as exc:
        return (), f"JSONDecodeError: {exc}"
    providers = tuple(str(provider) for provider in payload.get("available_providers") or [])
    error = str(payload.get("error") or "")
    if completed.returncode != 0 and not error:
        error = completed.stderr.strip() or f"probe_returncode={completed.returncode}"
    return providers, error


def build_suggestion_reason(entry: ModelRegistryEntry, failed_labels: Sequence[str]) -> str:
    if not failed_labels:
        return f"{entry.model_id} supports requested stems and has not been benchmarked yet"
    matched = sorted({label for label in failed_labels if label in set(entry.issue_labels_addressed)})
    if matched:
        return f"{entry.model_id} addresses failed labels: {', '.join(matched)}"
    return f"{entry.model_id} is a different model family for the requested stems"


def suggestion_sort_key(suggestion: ModelSuggestion) -> tuple[int, str, str]:
    available_rank = 0 if suggestion.availability.status == "available" else 1
    return (available_rank, suggestion.family, suggestion.model_id)


def default_model_registry(runtime_root: Path = DEFAULT_DOCKER_APP_ROOT) -> ModelRegistry:
    entries = (
        ModelRegistryEntry(
            model_id="htdemucs_ft",
            display_name="HTDemucs fine-tuned",
            family="demucs",
            target_stems=("vocals", "drums", "bass", "other", "instrumental"),
            output_stems=("vocals", "drums", "bass", "other", "instrumental"),
            command_template=(
                "python",
                "/app/tools/export_hq_stems_full.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "htdemucs_ft",
                "--device",
                "{device}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "low_sdr",
                "low_si_sdr",
                "source_conservation_error",
            ),
            aliases=("demucs_ft",),
            notes=("General-purpose baseline; model weights are resolved by Demucs/Torch cache.",),
            task_scope="four_stem",
            canonical_name_status="official_demucs_family",
            benchmark_score_context={
                "metric_family": "four_stem_average_sdr",
                "reported_score": "HTDemucs v4 fine-tuned avg SDR 9.20 in the source report",
                "directly_comparable_with": "four_stem_average",
                "comparability_warning": "Compare only against the same stem setup, data split, and extra-data/fine-tune regime.",
            },
            code_license="MIT",
            weights_license="upstream_demucs_weights",
            license_status="upstream_license_required",
            commercial_use_status="requires_weight_and_audio_rights_review",
            comparability_notes=(
                "Official Demucs/HTDemucs naming is canonical; community checkpoint labels must not be treated as separate paper-defined models.",
            ),
            report_alignment_notes=(
                "Source report recommends HTDemucs as the general-purpose pole for vocal/instrumental benchmarking.",
            ),
        ),
        ModelRegistryEntry(
            model_id="htdemucs_6s",
            display_name="HTDemucs 6-stem",
            family="demucs",
            target_stems=("vocals", "drums", "bass", "other", "guitar", "piano", "instrumental"),
            output_stems=("vocals", "drums", "bass", "other", "guitar", "piano", "instrumental"),
            command_template=(
                "python",
                "/app/tools/export_hq_stems_full.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "htdemucs_6s",
                "--device",
                "{device}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "harmonic_leakage_in_drums",
                "drum_tonal_leakage",
                "low_sdr",
                "low_si_sdr",
            ),
            aliases=("demucs_6s",),
            notes=("General 6-stem route; useful for drum/guitar/piano comparison candidates.",),
        ),
        ModelRegistryEntry(
            model_id="htdemucs_6s_aggressive_melody_first",
            display_name="HTDemucs 6-stem aggressive melody-first",
            family="demucs",
            target_stems=("vocals", "drums", "bass", "other", "guitar", "piano", "melody", "instrumental"),
            output_stems=("vocals", "drums", "bass", "other", "guitar", "piano", "melody", "instrumental"),
            command_template=(
                "python",
                "/app/tools/export_hq_stems_full.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "htdemucs_6s",
                "--device",
                "{device}",
                "--aggressive-melody-first",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "harmonic_leakage_in_drums",
                "drum_tonal_leakage",
                "non_target_leakage",
                "reference_stem_leakage",
                "low_sdr",
                "low_si_sdr",
            ),
            aliases=("demucs_6s_aggressive_melody_first", "htdemucs_6s_aggressive", "aggressive_melody_first_6s"),
            cost_class="high",
            notes=("Runs the generic aggressive melody-first rebalance to reduce harmonic leakage in drums.",),
        ),
        ModelRegistryEntry(
            model_id="tonal_harmonic_melody_v1",
            display_name="Tonal harmonic source-mix melody candidate",
            family="dsp",
            target_stems=("melody",),
            output_stems=("melody",),
            command_template=(
                "python",
                "/app/tools/build_tonal_harmonic_melody_candidate.py",
                "{source_audio_path}",
                "{output_dir}",
                "--listen-dir={listen_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "drum_tonal_leakage",
                "non_target_leakage",
                "reference_stem_leakage",
                "micro_reference_stem_leakage",
                "low_sdr",
                "low_si_sdr",
            ),
            aliases=("source_tonal_melody", "tonal_melody"),
            cost_class="medium",
            notes=(
                "No-reference source-mix STFT/HPSS tonal candidate for melody; "
                "must pass objective gate before delivery.",
            ),
        ),
        ModelRegistryEntry(
            model_id="pitch_salience_melody_v1",
            display_name="Pitch-salience reinforced melody candidate",
            family="hybrid",
            target_stems=("melody",),
            output_stems=("melody",),
            command_template=(
                "python",
                "/app/tools/build_pitch_salience_melody_candidate.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "hybrid_htdemucs_ft_bs_roformer_6s",
                "--device",
                "{device}",
                "--listen-dir={listen_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            issue_labels_addressed=(
                "melody_low_pitch_confidence",
                "reference_likeness_below_threshold",
                "vocal_residue_in_melody",
                "non_target_leakage",
                "reference_stem_leakage",
                "micro_reference_stem_leakage",
                "low_sdr",
                "low_si_sdr",
            ),
            aliases=("pitch_salience_melody", "melody_pitch_salience"),
            cost_class="high",
            notes=(
                "Builds a no-reference base melody from the source mix, then reinforces the adaptive "
                "dominant pitch/harmonic salience before the same listen and semantic gates.",
            ),
        ),
        ModelRegistryEntry(
            model_id="pitch_salience_melody_v2",
            display_name="Midband-safe pitch-salience melody candidate",
            family="hybrid",
            target_stems=("melody",),
            output_stems=("melody",),
            command_template=(
                "python",
                "/app/tools/build_pitch_salience_melody_candidate.py",
                "{source_audio_path}",
                "{output_dir}",
                "--route-name",
                "pitch_salience_melody_v2",
                "--model-name",
                "hybrid_htdemucs_ft_bs_roformer_6s",
                "--device",
                "{device}",
                "--min-frequency-hz",
                "120.0",
                "--max-harmonic-hz",
                "6000",
                "--bandwidth-ratio",
                "0.028",
                "--base-weight",
                "0.20",
                "--salience-weight",
                "1.10",
                "--salience-rms-gain",
                "0.85",
                "--listen-dir={listen_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            issue_labels_addressed=(
                "melody_bass_dominant",
                "melody_low_pitch_confidence",
                "reference_likeness_below_threshold",
                "vocal_residue_in_melody",
                "non_target_leakage",
                "reference_stem_leakage",
                "micro_reference_stem_leakage",
                "low_sdr",
                "low_si_sdr",
            ),
            aliases=("pitch_salience_melody_midband_safe", "melody_pitch_salience_midband_safe"),
            cost_class="high",
            notes=(
                "Uses the same registered pitch-salience builder with an explicit v2 route label "
                "and a 120 Hz minimum dominant-frequency search plus narrow midband salience "
                "to avoid bass-dominant melody acceptance.",
                "Must pass listen, GPU provenance, semantic judge, and stem matrix gates before delivery.",
            ),
        ),
        ModelRegistryEntry(
            model_id="model_stack_melody_v1",
            display_name="Model-stack debleeded melody candidate",
            family="hybrid",
            target_stems=("melody",),
            output_stems=("melody",),
            command_template=(
                "python",
                "/app/tools/build_model_stack_melody_candidate.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "htdemucs_6s",
                "--device",
                "{device}",
                "--listen-dir={listen_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "vocal_residue_in_melody",
                "non_target_leakage",
                "reference_stem_leakage",
                "micro_reference_stem_leakage",
                "drum_tonal_leakage",
                "low_sdr",
                "low_si_sdr",
            ),
            aliases=("stack_melody", "model_stack_melody", "melody_debleed_stack"),
            cost_class="high",
            notes=(
                "Runs a no-reference model-stack melody candidate and subtracts model-derived "
                "vocals/drums/bass projection before objective delivery gate.",
            ),
        ),
        ModelRegistryEntry(
            model_id="joint_vocal_melody_v1",
            display_name="Joint vocal/melody cross-debleed candidate",
            family="hybrid",
            target_stems=("vocals", "melody"),
            output_stems=("vocals", "melody"),
            command_template=(
                "python",
                "/app/tools/build_joint_vocal_melody_candidate.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "hybrid_htdemucs_ft_bs_roformer_6s",
                "--device",
                "{device}",
                "--listen-dir={listen_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "vocal_residue_in_melody",
                "melody_leakage_in_vocals",
                "cross_stem_leakage",
                "non_target_leakage",
                "reference_stem_leakage",
                "micro_reference_stem_leakage",
                "low_sdr",
                "low_si_sdr",
            ),
            aliases=("joint_vocal_melody", "vocal_melody_cross_debleed"),
            cost_class="high",
            notes=(
                "Builds separate no-reference vocals and melody candidates, then applies "
                "symmetric cross-projection debleed before objective delivery gate.",
            ),
        ),
        ModelRegistryEntry(
            model_id="joint_vocal_melody_v2",
            display_name="Joint vocal/melody source-pool partition debleed candidate",
            family="hybrid",
            target_stems=("vocals", "melody"),
            output_stems=("vocals", "melody"),
            command_template=(
                "python",
                "/app/tools/build_joint_vocal_melody_candidate_v2.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "hybrid_htdemucs_ft_bs_roformer_6s",
                "--device",
                "{device}",
                "--listen-dir={listen_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "exact_reference_threshold_failed",
                "sample_residual_above_threshold",
                "vocal_residue_in_melody",
                "melody_leakage_in_vocals",
                "cross_stem_leakage",
                "non_target_leakage",
                "reference_stem_leakage",
                "micro_reference_stem_leakage",
                "artifact_or_timbre_damage",
                "low_sdr",
                "low_si_sdr",
            ),
            aliases=("joint_vocal_melody_source_pool", "vocal_melody_source_pool_debleed"),
            cost_class="high",
            notes=(
                "Builds separate no-reference vocals and melody candidates from a non-target-suppressed "
                "source pool, then applies spectral partitioning plus projection debleed before objective gate.",
            ),
        ),
        ModelRegistryEntry(
            model_id="joint_vocal_melody_v3",
            display_name="Joint vocal/melody multi-candidate proxy router",
            family="hybrid",
            target_stems=("vocals", "melody"),
            output_stems=("vocals", "melody"),
            command_template=(
                "python",
                "/app/tools/build_joint_vocal_melody_candidate_v3.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "hybrid_htdemucs_ft_bs_roformer_6s",
                "--model-name",
                "htdemucs_6s",
                "--model-name",
                "htdemucs_ft",
                "--device",
                "{device}",
                "--listen-dir={listen_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "exact_reference_threshold_failed",
                "sample_residual_above_threshold",
                "vocal_residue_in_melody",
                "melody_leakage_in_vocals",
                "cross_stem_leakage",
                "non_target_leakage",
                "reference_stem_leakage",
                "micro_reference_stem_leakage",
                "artifact_or_timbre_damage",
                "low_sdr",
                "low_si_sdr",
            ),
            aliases=("joint_vocal_melody_multi_candidate", "vocal_melody_proxy_router"),
            cost_class="high",
            notes=(
                "Runs or consumes multiple no-reference model-stack candidates and selects vocals/melody "
                "independently by proxy leakage scores before objective and exact-reference gates.",
            ),
        ),
        ModelRegistryEntry(
            model_id="htdemucs_ft_drums_plus_percussive_fx_aux",
            display_name="HTDemucs fine-tuned drums + percussive FX aux",
            family="demucs",
            target_stems=("drums_plus_percussive_fx_aux",),
            output_stems=("drums_plus_percussive_fx_aux",),
            command_template=(
                "python",
                "/app/tools/export_drums_plus_percussive_fx_aux.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "htdemucs_ft",
                "--device",
                "{device}",
                "--listen-dir={listen_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "harmonic_leakage_in_drums",
                "drum_tonal_leakage",
                "bar_spanning_drop_melody_leakage",
                "non_target_leakage",
                "reference_stem_leakage",
                "micro_reference_stem_leakage",
                "low_sdr",
                "low_si_sdr",
            ),
            aliases=("demucs_ft_drums_plus_percussive_fx_aux",),
            cost_class="high",
            notes=(
                "No-reference fine-tuned Demucs route for the derived drums+percussive-FX target; "
                "keeps the same manifest contract as the 6-stem candidate route.",
            ),
        ),
        ModelRegistryEntry(
            model_id="htdemucs_6s_drums_plus_percussive_fx_aux",
            display_name="HTDemucs 6-stem drums + percussive FX aux",
            family="demucs",
            target_stems=("drums_plus_percussive_fx_aux",),
            output_stems=("drums_plus_percussive_fx_aux",),
            command_template=(
                "python",
                "/app/tools/export_drums_plus_percussive_fx_aux.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "htdemucs_6s",
                "--device",
                "{device}",
                "--listen-dir={listen_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "harmonic_leakage_in_drums",
                "drum_tonal_leakage",
                "bar_spanning_drop_melody_leakage",
                "non_target_leakage",
                "reference_stem_leakage",
                "micro_reference_stem_leakage",
                "low_sdr",
                "low_si_sdr",
            ),
            aliases=("drums_plus_percussive_fx_aux", "percussive_fx_aux_drums"),
            cost_class="high",
            notes=(
                "No-reference route for the derived drums+percussive-FX target; "
                "uses full-mix HTDemucs 6-stem aggressive melody-first export and relabels its drum bus.",
            ),
        ),
        ModelRegistryEntry(
            model_id="hybrid_htdemucs_ft_bs_roformer_6s_drums_plus_percussive_fx_aux",
            display_name="Hybrid HTDemucs + BS-RoFormer drums + percussive FX aux",
            family="hybrid",
            target_stems=("drums_plus_percussive_fx_aux",),
            output_stems=("drums_plus_percussive_fx_aux",),
            command_template=(
                "python",
                "/app/tools/export_drums_plus_percussive_fx_aux.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "hybrid_htdemucs_ft_bs_roformer_6s",
                "--device",
                "{device}",
                "--listen-dir={listen_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            required_paths=(
                "/app/vendor/music_source_separation_training",
                "/app/vendor/bs_roformer_official/config_bs_roformer_384_8_2_485100.yaml",
                "/app/vendor/bs_roformer_official/model_bs_roformer_ep_17_sdr_9.6568.ckpt",
            ),
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "harmonic_leakage_in_drums",
                "drum_tonal_leakage",
                "bar_spanning_drop_melody_leakage",
                "non_target_leakage",
                "reference_stem_leakage",
                "micro_reference_stem_leakage",
                "micro_residual_spike",
                "micro_transient_mismatch",
                "transient_or_attack_mismatch",
                "sample_residual_above_threshold",
                "low_sdr",
                "low_si_sdr",
            ),
            aliases=("hybrid_bs_drums_plus_percussive_fx_aux",),
            cost_class="high",
            notes=(
                "No-reference hybrid full-mix route for the derived drums+percussive-FX target; "
                "runs the BS-RoFormer + HTDemucs stack before the composite export contract is built.",
            ),
        ),
        ModelRegistryEntry(
            model_id="querysep_banquet_strings_orchestra_v1",
            display_name="QuerySep Banquet strings/orchestra",
            family="querysep",
            target_stems=("strings_orchestra",),
            output_stems=("strings_orchestra",),
            command_template=(
                "python",
                "/app/tools/build_querysep_strings_orchestra_candidate.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "htdemucs_6s",
                "--device",
                "{device}",
                "--listen-dir={listen_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "low_sdr",
                "low_si_sdr",
                "non_target_leakage",
                "reference_stem_leakage",
                "micro_reference_stem_leakage",
                "possible_other_proxy_leakage",
                "long_tail_presence_too_weak",
                "long_tail_active_presence_too_sparse",
                "long_tail_presence_too_weak_vs_source",
                "sample_residual_above_threshold",
                "spectral_timbre_mismatch",
                "target_shape_mismatch",
            ),
            aliases=("querysep_strings_orchestra", "banquet_strings_orchestra"),
            cost_class="high",
            notes=(
                "No-reference long-tail route: builds base no-reference stems from the source mix, "
                "then calls querysep-worker with Banquet BYOQ and the long-tail acoustic quality gate.",
            ),
        ),
        ModelRegistryEntry(
            model_id="model_stack_strings_orchestra_v1",
            display_name="Model-stack strings/orchestra harmonic residual route",
            family="hybrid",
            target_stems=("strings_orchestra",),
            output_stems=("strings_orchestra",),
            command_template=(
                "python",
                "/app/tools/build_model_stack_strings_orchestra_candidate.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "htdemucs_6s",
                "--device",
                "{device}",
                "--listen-dir={listen_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "low_sdr",
                "low_si_sdr",
                "non_target_leakage",
                "reference_stem_leakage",
                "micro_reference_stem_leakage",
                "micro_residual_reference_stem_leakage",
                "possible_other_proxy_leakage",
                "long_tail_presence_too_weak",
                "long_tail_active_presence_too_sparse",
                "long_tail_presence_too_weak_vs_source",
                "sample_residual_above_threshold",
                "spectral_timbre_mismatch",
                "target_shape_mismatch",
            ),
            aliases=("model_stack_strings_orchestra", "harmonic_residual_strings_orchestra"),
            cost_class="high",
            notes=(
                "No-reference strings/orchestra candidate: builds a source-mix model stack, combines "
                "harmonic other/melody/piano/guitar evidence, and projection-debleeds vocals/drums/bass/melody.",
            ),
        ),
        ModelRegistryEntry(
            model_id="model_stack_strings_orchestra_hybrid_bs_v1",
            display_name="Model-stack strings/orchestra route with hybrid BS-RoFormer base",
            family="hybrid",
            target_stems=("strings_orchestra",),
            output_stems=("strings_orchestra",),
            command_template=(
                "python",
                "/app/tools/build_model_stack_strings_orchestra_candidate.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "hybrid_htdemucs_ft_bs_roformer_6s",
                "--device",
                "{device}",
                "--route-name",
                "model_stack_strings_orchestra_hybrid_bs_v1",
                "--listen-dir={listen_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            required_paths=(
                "/app/vendor/music_source_separation_training",
                "/app/vendor/bs_roformer_official/config_bs_roformer_384_8_2_485100.yaml",
                "/app/vendor/bs_roformer_official/model_bs_roformer_ep_17_sdr_9.6568.ckpt",
            ),
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "low_sdr",
                "low_si_sdr",
                "non_target_leakage",
                "reference_stem_leakage",
                "micro_reference_stem_leakage",
                "micro_residual_reference_stem_leakage",
                "possible_other_proxy_leakage",
                "long_tail_presence_too_weak",
                "long_tail_active_presence_too_sparse",
                "long_tail_presence_too_weak_vs_source",
                "sample_residual_above_threshold",
                "spectral_timbre_mismatch",
                "target_shape_mismatch",
                "waveform_shape_mismatch",
            ),
            aliases=("model_stack_strings_orchestra_hybrid_bs", "bs_harmonic_residual_strings_orchestra"),
            cost_class="high",
            notes=(
                "No-reference strings/orchestra candidate: uses the existing model-stack strings builder "
                "with the stronger hybrid HTDemucs + BS-RoFormer source-mix base before objective delivery gate.",
            ),
        ),
        ModelRegistryEntry(
            model_id="scnet_xl_ihf_musdb18",
            display_name="SCNet XL IHF MUSDB18 checkpoint",
            family="scnet",
            target_stems=("drums", "bass", "other", "vocals"),
            output_stems=("drums", "bass", "other", "vocals"),
            command_template=(
                "python",
                "/app/tools/export_scnet_full.py",
                "{source_audio_path}",
                "{output_dir}",
                "--device-id",
                "0",
                "--num-overlap",
                "4",
                "--listen-dir={listen_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            required_paths=(
                "/app/vendor/music_source_separation_training",
                "/app/vendor/scnet_xl_ihf/config_musdb18_scnet_xl_more_wide_v5.yaml",
                "/app/vendor/scnet_xl_ihf/model_scnet_ep_36_sdr_10.0891.ckpt",
            ),
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "harmonic_leakage_in_drums",
                "drum_tonal_leakage",
                "non_target_leakage",
                "reference_stem_leakage",
                "low_sdr",
                "low_si_sdr",
                "micro_reference_stem_leakage",
                "micro_residual_spike",
                "micro_transient_mismatch",
                "transient_or_attack_mismatch",
                "sample_residual_above_threshold",
                "waveform_shape_mismatch",
            ),
            aliases=("scnet", "scnet_xl", "scnet_ihf"),
            cost_class="high",
            notes=(
                "External SCNet full-track multi-stem candidate for drums/bass/other/vocals; "
                "must be benchmark-gated before delivery.",
            ),
            task_scope="four_stem",
            canonical_name_status="paper_defined_model_family",
            benchmark_score_context={
                "metric_family": "four_stem_average_sdr",
                "reported_score": "SCNet avg SDR 9.0 on MUSDB18-HQ without extra-data in the source report",
                "directly_comparable_with": "four_stem_average",
                "comparability_warning": "Do not compare this number directly with vocal-only SDR checkpoints.",
            },
            code_license="MIT",
            weights_license="checkpoint_specific",
            license_status="checkpoint_license_unverified",
            commercial_use_status="requires_checkpoint_rights_review",
            comparability_notes=(
                "Four-stem average SDR is not directly comparable with vocal-only Mel/RoFormer numbers.",
            ),
            report_alignment_notes=(
                "Source report highlights SCNet as a modern efficient full-stem baseline.",
            ),
        ),
        ModelRegistryEntry(
            model_id="hybrid_htdemucs_ft_bs_roformer_6s",
            display_name="Hybrid HTDemucs + BS-RoFormer + 6-stem",
            family="hybrid",
            target_stems=("vocals", "drums", "bass", "other", "guitar", "piano", "instrumental"),
            output_stems=("vocals", "drums", "bass", "other", "guitar", "piano", "instrumental"),
            command_template=(
                "python",
                "/app/tools/export_hq_stems_full.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "hybrid_htdemucs_ft_bs_roformer_6s",
                "--device",
                "{device}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            required_paths=(
                "/app/vendor/music_source_separation_training",
                "/app/vendor/bs_roformer_official/config_bs_roformer_384_8_2_485100.yaml",
                "/app/vendor/bs_roformer_official/model_bs_roformer_ep_17_sdr_9.6568.ckpt",
            ),
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "vocal_residue_in_minus",
                "vocal_dropout",
                "vocal_tail_loss",
                "harmonic_leakage_in_drums",
            ),
            aliases=("bs_roformer", "hybrid_bs"),
            cost_class="high",
            notes=("Uses BS-RoFormer vocals/other plus Demucs stages; requires local vendor checkpoint.",),
        ),
        ModelRegistryEntry(
            model_id="hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first",
            display_name="Hybrid HTDemucs + BS-RoFormer + 6-stem aggressive melody-first",
            family="hybrid",
            target_stems=("vocals", "drums", "bass", "other", "guitar", "piano", "melody", "instrumental"),
            output_stems=("vocals", "drums", "bass", "other", "guitar", "piano", "melody", "instrumental"),
            command_template=(
                "python",
                "/app/tools/export_hq_stems_full.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-name",
                "hybrid_htdemucs_ft_bs_roformer_6s",
                "--device",
                "{device}",
                "--aggressive-melody-first",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            required_paths=(
                "/app/vendor/music_source_separation_training",
                "/app/vendor/bs_roformer_official/config_bs_roformer_384_8_2_485100.yaml",
                "/app/vendor/bs_roformer_official/model_bs_roformer_ep_17_sdr_9.6568.ckpt",
            ),
            issue_labels_addressed=(
                "reference_likeness_below_threshold",
                "vocal_residue_in_minus",
                "vocal_dropout",
                "vocal_tail_loss",
                "harmonic_leakage_in_drums",
                "drum_tonal_leakage",
                "non_target_leakage",
                "reference_stem_leakage",
                "low_sdr",
                "low_si_sdr",
            ),
            aliases=("hybrid_bs_aggressive_melody_first", "aggressive_hybrid_bs"),
            cost_class="high",
            notes=("Hybrid BS route plus generic aggressive melody-first rebalance for leaked harmonic buses.",),
        ),
        ModelRegistryEntry(
            model_id="mdx23c_vocals",
            display_name="MDX23C vocal checkpoint",
            family="mdx",
            target_stems=("vocals", "instrumental"),
            output_stems=("vocals", "instrumental"),
            command_template=(
                "python",
                "/app/tools/export_mdx23c_vocal_full.py",
                "{source_audio_path}",
                "{output_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            required_paths=(
                "/app/vendor/mdx23c_vocals_official/config_vocals_mdx23c.yaml",
                "/app/vendor/mdx23c_vocals_official/model_vocals_mdx23c_sdr_10.17.ckpt",
            ),
            issue_labels_addressed=(
                "vocal_dropout",
                "vocal_tail_loss",
                "presence_loss",
                "thin_or_overcleaned",
                "reference_likeness_below_threshold",
            ),
            aliases=("mdx23c", "mdx"),
            notes=("External vocal candidate family; benchmark must verify before delivery.",),
        ),
        ModelRegistryEntry(
            model_id="melband_roformer_vocals",
            display_name="MelBand RoFormer vocal checkpoint",
            family="roformer",
            target_stems=("vocals", "instrumental"),
            output_stems=("vocals", "instrumental"),
            command_template=(
                "python",
                "/app/tools/export_melband_vocal_full.py",
                "{source_audio_path}",
                "{output_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            required_paths=(
                "/app/vendor/mel_band_roformer_vocal_model",
                "/app/vendor/mel_band_roformer_vocal_model/configs/config_vocals_mel_band_roformer.yaml",
                "/app/vendor/mel_band_roformer_vocal_model_weights/MelBandRoformer.ckpt",
            ),
            issue_labels_addressed=(
                "vocal_dropout",
                "vocal_tail_loss",
                "presence_loss",
                "vocal_harshness",
                "high_noise_or_robotic_artifact",
                "reference_likeness_below_threshold",
            ),
            aliases=("melband", "mel_band_roformer"),
            cost_class="high",
            notes=("External RoFormer vocal route with DSP cleanup variants.",),
            task_scope="vocal_instrumental",
            canonical_name_status="paper_family_with_project_checkpoint",
            benchmark_score_context={
                "metric_family": "vocal_only_sdr",
                "reported_score": "Mel-RoFormer vocals SDR 12.08 MUSDB18HQ-only / 13.29 with extra-data in the source report",
                "directly_comparable_with": "vocal_only",
                "comparability_warning": "Vocal-only SDR is not directly comparable with four-stem average SDR.",
            },
            code_license="wrapper_mit",
            weights_license="checkpoint_specific",
            license_status="checkpoint_license_unverified",
            commercial_use_status="requires_checkpoint_rights_review",
            comparability_notes=(
                "Mel/RoFormer vocal-only results are not directly comparable with HTDemucs or SCNet four-stem averages.",
                "Use this route as a vocal-specialized pole in the model zoo, not as a universal leaderboard winner.",
            ),
            report_alignment_notes=(
                "Source report identifies Mel-RoFormer as the practical vocal/instrumental specialist.",
            ),
        ),
        ModelRegistryEntry(
            model_id="becruily_karaoke_vocals",
            display_name="Becruily MelBand RoFormer karaoke checkpoint",
            family="roformer",
            target_stems=("vocals", "instrumental"),
            output_stems=("vocals", "instrumental"),
            command_template=(
                "python",
                "/app/tools/export_melband_vocal_full.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-path",
                "/app/vendor/becruily_karaoke/mel_band_roformer_karaoke_becruily.ckpt",
                "--config-path",
                "/app/vendor/becruily_karaoke/config_karaoke_becruily.yaml",
                "--manifest-model-id",
                "becruily_karaoke_vocals",
                "--manifest-model-display-name",
                "Becruily MelBand RoFormer karaoke checkpoint",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            required_paths=(
                "/app/vendor/mel_band_roformer_vocal_model",
                "/app/vendor/becruily_karaoke/config_karaoke_becruily.yaml",
                "/app/vendor/becruily_karaoke/mel_band_roformer_karaoke_becruily.ckpt",
            ),
            issue_labels_addressed=(
                "vocal_dropout",
                "vocal_tail_loss",
                "presence_loss",
                "vocal_harshness",
                "reference_likeness_below_threshold",
                "target_shape_mismatch",
            ),
            aliases=("becruily_karaoke", "karaoke_becruily"),
            cost_class="high",
            notes=("Uses the shared MelBand exporter with config-schema normalization for karaoke checkpoints.",),
        ),
        ModelRegistryEntry(
            model_id="becruily_instrumental_vocals",
            display_name="Becruily MelBand RoFormer instrumental checkpoint",
            family="roformer",
            target_stems=("vocals", "instrumental"),
            output_stems=("vocals", "instrumental"),
            command_template=(
                "python",
                "/app/tools/export_melband_vocal_full.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-path",
                "/app/vendor/becruily_instrumental/mel_band_roformer_instrumental_becruily.ckpt",
                "--config-path",
                "/app/vendor/becruily_instrumental/config_instrumental_becruily.yaml",
                "--manifest-model-id",
                "becruily_instrumental_vocals",
                "--manifest-model-display-name",
                "Becruily MelBand RoFormer instrumental checkpoint",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            required_paths=(
                "/app/vendor/mel_band_roformer_vocal_model",
                "/app/vendor/becruily_instrumental/config_instrumental_becruily.yaml",
                "/app/vendor/becruily_instrumental/mel_band_roformer_instrumental_becruily.ckpt",
            ),
            issue_labels_addressed=(
                "vocal_residue_in_minus",
                "wide_stereo_non_vocal_leakage",
                "reference_likeness_below_threshold",
                "target_shape_mismatch",
                "vocal_dropout",
                "vocal_tail_loss",
                "presence_loss",
            ),
            aliases=("becruily_instrumental", "instrumental_becruily"),
            cost_class="high",
            notes=(
                "Uses the shared MelBand exporter with config-schema normalization for the Becruily instrumental checkpoint.",
                "Registered primarily as an instrumental/minus candidate; the companion vocal stem must still pass mode-specific gates before delivery.",
            ),
            task_scope="vocal_instrumental",
            canonical_name_status="community_checkpoint",
            benchmark_score_context={
                "metric_family": "vocal_instrumental_checkpoint_sdr",
                "reported_score": "No trusted local score yet; this route is a candidate for local no-reference and exact-reference gates",
                "directly_comparable_with": "same_task_local_benchmark",
                "comparability_warning": "Community checkpoint labels are provenance metadata, not production quality claims.",
            },
            code_license="wrapper_mit",
            weights_license="checkpoint_specific",
            license_status="checkpoint_license_unverified",
            commercial_use_status="requires_checkpoint_rights_review",
            comparability_notes=(
                "Instrumental-focused checkpoints are local candidates for minus preservation, not release-ready artifacts by registration alone.",
            ),
            report_alignment_notes=(
                "Adds a reproducible MelBand/MSST candidate that can run in the existing GPU Docker image without the audio-separator venv.",
            ),
        ),
        ModelRegistryEntry(
            model_id="bs_mamba2_vocals",
            display_name="BS-Mamba2 vocal checkpoint",
            family="mamba",
            target_stems=("vocals", "instrumental"),
            output_stems=("vocals", "instrumental"),
            command_template=(
                "python",
                "/app/tools/export_bs_mamba2_vocal_full.py",
                "{source_audio_path}",
                "{output_dir}",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            required_paths=(
                "/app/vendor/bs_mamba2_vocals/config_bs_mamba2_vocals.yaml",
                "/app/vendor/bs_mamba2_vocals/bs_mamba2_vocals.ckpt",
                "/app/vendor/bs_mamba2_vocals/runtime_probe_passed.json",
            ),
            required_python_modules=("mamba_ssm", "causal_conv1d"),
            issue_labels_addressed=(
                "vocal_dropout",
                "vocal_tail_loss",
                "presence_loss",
                "reference_likeness_below_threshold",
            ),
            aliases=("bs_mamba2",),
            cost_class="high",
            notes=(
                "Optional external candidate; availability depends on local checkpoint/runtime.",
                "Requires an explicit runtime_probe_passed.json marker because module presence alone has reached model load and then SIGSEGVed before manifest creation.",
            ),
        ),
        ModelRegistryEntry(
            model_id="audio_separator_bs_roformer_vocals",
            display_name="Audio Separator BS-RoFormer vocal checkpoint",
            family="audio_separator_uvr",
            target_stems=("vocals", "instrumental"),
            output_stems=("vocals", "instrumental"),
            command_template=(
                "python",
                "/app/tools/export_audio_separator_vocal_full.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-filename",
                "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                "--model-file-dir",
                "/app/models/audio_separator",
                "--audio-separator-python",
                "/opt/audio-separator-venv/bin/python",
                "--manifest-model-id",
                "audio_separator_bs_roformer_vocals",
                "--manifest-model-display-name",
                "Audio Separator BS-RoFormer vocal checkpoint",
                "--require-cuda-provider",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            required_paths=(
                "/opt/audio-separator-venv/bin/python",
                "/app/models/audio_separator/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                "/app/models/audio_separator/model_bs_roformer_ep_317_sdr_12.9755.yaml",
            ),
            issue_labels_addressed=(
                "vocal_dropout",
                "vocal_tail_loss",
                "presence_loss",
                "thin_or_overcleaned",
                "high_noise_or_robotic_artifact",
                "kick_punch_bleed",
                "wide_stereo_non_vocal_leakage",
                "reference_likeness_below_threshold",
            ),
            aliases=("audio_separator_bs_roformer", "uvr_bs_roformer_vocals"),
            cost_class="high",
            notes=(
                "Audio Separator UVR route using BS-RoFormer; registered candidate only, "
                "must pass benchmark/effectiveness gate before delivery.",
            ),
            task_scope="vocal_instrumental",
            canonical_name_status="community_checkpoint",
            benchmark_score_context={
                "metric_family": "vocal_checkpoint_sdr",
                "reported_score": "Checkpoint filename reports SDR 12.9755; verify on local domain before claims",
                "directly_comparable_with": "same_task_local_benchmark",
                "comparability_warning": "Community checkpoint score is provenance metadata, not a production claim.",
            },
            code_license="audio_separator_runtime",
            weights_license="checkpoint_specific",
            license_status="checkpoint_license_unverified",
            commercial_use_status="requires_checkpoint_rights_review",
            comparability_notes=(
                "Community checkpoint labels are not paper-defined model names and must be benchmarked locally.",
            ),
            report_alignment_notes=(
                "Source report treats BS/RoFormer-style routes as strong candidates, but local exact-reference gates decide delivery.",
            ),
        ),
        ModelRegistryEntry(
            model_id="audio_separator_bs_roformer_viperx_1297_vendor_config_vocals",
            display_name="Audio Separator BS-RoFormer ViperX-1297 vendor config variant",
            family="audio_separator_uvr",
            target_stems=("vocals", "instrumental"),
            output_stems=("vocals", "instrumental"),
            command_template=(
                "python",
                "/app/tools/export_audio_separator_vocal_full.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-filename",
                "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                "--model-file-dir",
                "/app/vendor/bs_roformer_viperx_1297",
                "--audio-separator-python",
                "/opt/audio-separator-venv/bin/python",
                "--manifest-model-id",
                "audio_separator_bs_roformer_viperx_1297_vendor_config_vocals",
                "--manifest-model-display-name",
                "Audio Separator BS-RoFormer ViperX-1297 vendor config variant",
                "--require-cuda-provider",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            required_paths=(
                "/opt/audio-separator-venv/bin/python",
                "/app/vendor/bs_roformer_viperx_1297/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                "/app/vendor/bs_roformer_viperx_1297/model_bs_roformer_ep_317_sdr_12.9755.yaml",
            ),
            issue_labels_addressed=(
                "vocal_dropout",
                "vocal_tail_loss",
                "presence_loss",
                "thin_or_overcleaned",
                "high_noise_or_robotic_artifact",
                "kick_punch_bleed",
                "wide_stereo_non_vocal_leakage",
                "reference_likeness_below_threshold",
            ),
            aliases=("audio_separator_bs_roformer_viperx_1297_vendor_config", "uvr_bs_roformer_viperx_1297_vendor_config"),
            cost_class="high",
            notes=(
                "Uses the same ViperX-1297 checkpoint as audio_separator_bs_roformer_vocals with the vendor sidecar YAML as a config variant.",
                "Registered candidate only; must pass mode-specific judge, selector, and artifact contract before any listen delivery.",
            ),
            task_scope="vocal_instrumental",
            canonical_name_status="community_checkpoint_config_variant",
            benchmark_score_context={
                "metric_family": "vocal_checkpoint_sdr",
                "reported_score": "Checkpoint filename reports SDR 12.9755; this route varies only the local sidecar config",
                "directly_comparable_with": "same_task_local_benchmark",
                "comparability_warning": "Same-checkpoint config variants are candidates for local judging, not separate leaderboard claims.",
            },
            code_license="audio_separator_runtime",
            weights_license="checkpoint_specific",
            license_status="checkpoint_license_unverified",
            commercial_use_status="requires_checkpoint_rights_review",
            comparability_notes=(
                "This route is a config variant over the same checkpoint, not a new model-family result.",
                "Community checkpoint labels are not paper-defined model names and must be benchmarked locally.",
            ),
            report_alignment_notes=(
                "Adds a reproducible sidecar-config variation to the local candidate matrix without relaxing gates.",
            ),
        ),
        ModelRegistryEntry(
            model_id="audio_separator_melband_roformer_viperx_vocals",
            display_name="Audio Separator MelBand RoFormer ViperX vocal checkpoint",
            family="audio_separator_uvr",
            target_stems=("vocals", "instrumental"),
            output_stems=("vocals", "instrumental"),
            command_template=(
                "python",
                "/app/tools/export_audio_separator_vocal_full.py",
                "{source_audio_path}",
                "{output_dir}",
                "--model-filename",
                "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
                "--model-file-dir",
                "/app/vendor/melband_roformer_viperx_11436",
                "--audio-separator-python",
                "/opt/audio-separator-venv/bin/python",
                "--manifest-model-id",
                "audio_separator_melband_roformer_viperx_vocals",
                "--manifest-model-display-name",
                "Audio Separator MelBand RoFormer ViperX vocal checkpoint",
                "--require-cuda-provider",
            ),
            expected_manifest="{output_dir}/stem_manifest.json",
            required_paths=(
                "/opt/audio-separator-venv/bin/python",
                "/app/vendor/melband_roformer_viperx_11436/model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
                "/app/vendor/melband_roformer_viperx_11436/model_mel_band_roformer_ep_3005_sdr_11.4360.yaml",
            ),
            issue_labels_addressed=(
                "vocal_dropout",
                "vocal_tail_loss",
                "presence_loss",
                "vocal_harshness",
                "high_noise_or_robotic_artifact",
                "drop_melody_leakage",
                "kick_punch_bleed",
                "wide_stereo_non_vocal_leakage",
                "reference_likeness_below_threshold",
            ),
            aliases=("audio_separator_melband_roformer", "uvr_melband_roformer_viperx_vocals"),
            cost_class="high",
            notes=(
                "Audio Separator UVR route using the ViperX MelBand RoFormer vocal checkpoint; "
                "registered candidate only, must pass RALPH-AI gate before delivery.",
            ),
            task_scope="vocal_instrumental",
            canonical_name_status="community_checkpoint",
            benchmark_score_context={
                "metric_family": "vocal_checkpoint_sdr",
                "reported_score": "Checkpoint filename reports SDR 11.4360; verify on local domain before claims",
                "directly_comparable_with": "same_task_local_benchmark",
                "comparability_warning": "Community checkpoint score is provenance metadata, not a production claim.",
            },
            code_license="audio_separator_runtime",
            weights_license="checkpoint_specific",
            license_status="checkpoint_license_unverified",
            commercial_use_status="requires_checkpoint_rights_review",
            comparability_notes=(
                "Vocal-only/community checkpoint results are not directly comparable with four-stem average SDR.",
            ),
            report_alignment_notes=(
                "Source report recommends Mel/RoFormer-style vocal specialists as model-zoo candidates under license review.",
            ),
        ),
    )
    return ModelRegistry(entries=entries, runtime_root=runtime_root)
