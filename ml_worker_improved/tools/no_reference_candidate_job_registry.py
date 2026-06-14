from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_ml_dataset import stable_hash  # noqa: E402
from worker.model_registry import ModelRegistry, default_model_registry  # noqa: E402


REGISTERED_MODEL_ENTRYPOINT = "tools/run_registered_model_candidate.py"
DOCKER_REGISTERED_MODEL_ENTRYPOINT = "/app/tools/run_registered_model_candidate.py"
ALLOWED_PROFILES = ("safe", "balanced", "exploration")
ALLOWED_TARGET_MODES = ("vocal", "minus", "both")
ALLOWED_FAMILIES = (
    "balanced_pair_default",
    "vocal_focused_default",
    "minus_focused_default",
    "artifact_safe_default",
    "vocal_focused",
    "minus_focused",
    "balanced_pair",
    "artifact_safe",
    "chunk_overlap_variant",
    "high_overlap_variant",
    "low_overlap_variant",
)
ALLOWED_ENTRYPOINT_TYPES = ("python_module", "python_script", "existing_cli", "unavailable")


@dataclass(frozen=True)
class CandidateJobConfig:
    candidate_config_id: str
    candidate_family: str
    target_mode: str
    source_model_id: str
    source_tool: str
    entrypoint_type: str
    command_template: tuple[str, ...]
    expected_outputs: dict[str, str]
    gpu_required: bool
    timeout_sec: int
    max_retries: int
    profile_allowed: tuple[str, ...]
    safety_notes: tuple[str, ...]
    supported_parameters: tuple[str, ...] = ()
    shared_execution_group_id: str = ""
    produces_vocal: bool = False
    produces_minus: bool = False
    produces_both: bool = False
    materialized_candidate_modes: tuple[str, ...] = ()
    reuse_outputs_across_modes: bool = False
    target_stem: str = ""
    source_config_hash: str = ""

    def __post_init__(self) -> None:
        if self.target_mode not in ALLOWED_TARGET_MODES:
            raise ValueError(f"Unsupported target_mode: {self.target_mode}")
        if self.candidate_family not in ALLOWED_FAMILIES:
            raise ValueError(f"Unsupported candidate_family: {self.candidate_family}")
        if self.entrypoint_type not in ALLOWED_ENTRYPOINT_TYPES:
            raise ValueError(f"Unsupported entrypoint_type: {self.entrypoint_type}")
        if not self.shared_execution_group_id:
            object.__setattr__(self, "shared_execution_group_id", self.candidate_config_id)
        if not self.materialized_candidate_modes:
            if self.target_mode == "both":
                object.__setattr__(self, "materialized_candidate_modes", ("vocal", "minus", "both"))
            else:
                object.__setattr__(self, "materialized_candidate_modes", (self.target_mode,))
        if self.produces_vocal and self.produces_minus and not self.produces_both:
            object.__setattr__(self, "produces_both", True)
        if not self.source_config_hash:
            object.__setattr__(self, "source_config_hash", stable_hash(self.to_manifest(include_hash=False)))

    def to_manifest(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "candidate_config_id": self.candidate_config_id,
            "candidate_family": self.candidate_family,
            "target_mode": self.target_mode,
            "source_model_id": self.source_model_id,
            "source_tool": self.source_tool,
            "entrypoint_type": self.entrypoint_type,
            "command_template": list(self.command_template),
            "expected_outputs": dict(self.expected_outputs),
            "gpu_required": bool(self.gpu_required),
            "timeout_sec": int(self.timeout_sec),
            "max_retries": int(self.max_retries),
            "profile_allowed": list(self.profile_allowed),
            "safety_notes": list(self.safety_notes),
            "supported_parameters": list(self.supported_parameters),
            "shared_execution_group_id": self.shared_execution_group_id,
            "produces_vocal": bool(self.produces_vocal),
            "produces_minus": bool(self.produces_minus),
            "produces_both": bool(self.produces_both),
            "materialized_candidate_modes": list(self.materialized_candidate_modes),
            "reuse_outputs_across_modes": bool(self.reuse_outputs_across_modes),
            "target_stem": self.target_stem,
        }
        if include_hash:
            payload["source_config_hash"] = self.source_config_hash
        return payload


def registered_model_command_template() -> tuple[str, ...]:
    return (
        "python",
        DOCKER_REGISTERED_MODEL_ENTRYPOINT,
        "{input_audio}",
        "{registered_output_dir}",
        "--model-id",
        "{source_model_id}",
        "--device",
        "cuda",
        "--target-stem",
        "{target_stem}",
    )


def registered_model_supported_parameters() -> tuple[str, ...]:
    return (
        "source_audio_path",
        "output_dir",
        "model_id",
        "device",
        "target_stem",
    )


def local_registered_entrypoint_exists() -> bool:
    return (APP_ROOT / REGISTERED_MODEL_ENTRYPOINT).exists()


def model_can_run(registry: ModelRegistry, model_id: str) -> tuple[bool, tuple[str, ...]]:
    try:
        entry = registry.get(model_id)
    except KeyError:
        return False, ("model_id_missing_from_registry",)
    availability = registry.availability(entry)
    if availability.status == "available":
        return True, ()
    notes = []
    notes.extend(f"missing_path:{path}" for path in availability.missing_paths)
    notes.extend(f"missing_python_module:{module}" for module in availability.missing_python_modules)
    notes.extend(f"runtime_blocker:{blocker}" for blocker in availability.runtime_blockers)
    return False, tuple(notes or ("model_unavailable",))


def config_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "model_id": "htdemucs_ft",
            "target_mode": "both",
            "target_stem": "minus",
            "family": "balanced_pair_default",
            "profiles": ("safe", "balanced", "exploration"),
            "timeout_sec": 900,
            "variant_suffix": "shared",
            "shared_execution_group_id": "local_registered__htdemucs_ft__default",
            "produces_vocal": True,
            "produces_minus": True,
            "materialized_candidate_modes": ("vocal", "minus", "both"),
            "reuse_outputs_across_modes": True,
        },
        {
            "model_id": "htdemucs_ft",
            "target_mode": "vocal",
            "target_stem": "vocal",
            "family": "vocal_focused_default",
            "profiles": ("safe", "balanced", "exploration"),
            "timeout_sec": 900,
            "shared_execution_group_id": "local_registered__htdemucs_ft__default",
            "produces_vocal": True,
            "produces_minus": False,
            "materialized_candidate_modes": ("vocal",),
            "reuse_outputs_across_modes": True,
        },
        {
            "model_id": "htdemucs_ft",
            "target_mode": "minus",
            "target_stem": "minus",
            "family": "minus_focused_default",
            "profiles": ("safe", "balanced", "exploration"),
            "timeout_sec": 900,
            "shared_execution_group_id": "local_registered__htdemucs_ft__default",
            "produces_vocal": False,
            "produces_minus": True,
            "materialized_candidate_modes": ("minus",),
            "reuse_outputs_across_modes": True,
        },
        {
            "model_id": "htdemucs_6s",
            "target_mode": "both",
            "target_stem": "minus",
            "family": "artifact_safe_default",
            "profiles": ("balanced", "exploration"),
            "timeout_sec": 900,
            "variant_suffix": "shared",
            "shared_execution_group_id": "local_registered__htdemucs_6s__default",
            "produces_vocal": True,
            "produces_minus": True,
            "materialized_candidate_modes": ("vocal", "minus", "both"),
            "reuse_outputs_across_modes": True,
        },
        {
            "model_id": "htdemucs_6s",
            "target_mode": "vocal",
            "target_stem": "vocal",
            "family": "artifact_safe_default",
            "profiles": ("balanced", "exploration"),
            "timeout_sec": 900,
            "shared_execution_group_id": "local_registered__htdemucs_6s__default",
            "produces_vocal": True,
            "produces_minus": False,
            "materialized_candidate_modes": ("vocal",),
            "reuse_outputs_across_modes": True,
        },
        {
            "model_id": "htdemucs_6s",
            "target_mode": "minus",
            "target_stem": "minus",
            "family": "artifact_safe_default",
            "profiles": ("balanced", "exploration"),
            "timeout_sec": 900,
            "shared_execution_group_id": "local_registered__htdemucs_6s__default",
            "produces_vocal": False,
            "produces_minus": True,
            "materialized_candidate_modes": ("minus",),
            "reuse_outputs_across_modes": True,
        },
        {
            "model_id": "scnet_xl_ihf_musdb18",
            "target_mode": "vocal",
            "target_stem": "vocal",
            "family": "artifact_safe_default",
            "profiles": ("exploration",),
            "timeout_sec": 1200,
        },
        {
            "model_id": "mdx23c_vocals",
            "target_mode": "vocal",
            "target_stem": "vocal",
            "family": "vocal_focused_default",
            "profiles": ("exploration",),
            "timeout_sec": 1200,
        },
        {
            "model_id": "melband_roformer_vocals",
            "target_mode": "vocal",
            "target_stem": "vocal",
            "family": "vocal_focused_default",
            "profiles": ("exploration",),
            "timeout_sec": 1200,
        },
        {
            "model_id": "htdemucs_ft",
            "target_mode": "vocal",
            "target_stem": "vocal",
            "family": "chunk_overlap_variant",
            "profiles": ("balanced", "exploration"),
            "timeout_sec": 900,
            "variant_suffix": "chunk_overlap",
            "shared_execution_group_id": "local_registered__htdemucs_ft__chunk_overlap",
            "unsupported_parameters": ("chunk_size", "overlap"),
        },
        {
            "model_id": "htdemucs_ft",
            "target_mode": "minus",
            "target_stem": "minus",
            "family": "chunk_overlap_variant",
            "profiles": ("balanced", "exploration"),
            "timeout_sec": 900,
            "variant_suffix": "chunk_overlap",
            "shared_execution_group_id": "local_registered__htdemucs_ft__chunk_overlap",
            "unsupported_parameters": ("chunk_size", "overlap"),
        },
        {
            "model_id": "htdemucs_ft",
            "target_mode": "vocal",
            "target_stem": "vocal",
            "family": "high_overlap_variant",
            "profiles": ("exploration",),
            "timeout_sec": 900,
            "variant_suffix": "high_overlap",
            "shared_execution_group_id": "local_registered__htdemucs_ft__high_overlap",
            "unsupported_parameters": ("overlap",),
        },
        {
            "model_id": "htdemucs_ft",
            "target_mode": "minus",
            "target_stem": "minus",
            "family": "high_overlap_variant",
            "profiles": ("exploration",),
            "timeout_sec": 900,
            "variant_suffix": "high_overlap",
            "shared_execution_group_id": "local_registered__htdemucs_ft__high_overlap",
            "unsupported_parameters": ("overlap",),
        },
        {
            "model_id": "htdemucs_ft",
            "target_mode": "vocal",
            "target_stem": "vocal",
            "family": "low_overlap_variant",
            "profiles": ("exploration",),
            "timeout_sec": 900,
            "variant_suffix": "low_overlap",
            "shared_execution_group_id": "local_registered__htdemucs_ft__low_overlap",
            "unsupported_parameters": ("overlap",),
        },
        {
            "model_id": "htdemucs_ft",
            "target_mode": "minus",
            "target_stem": "minus",
            "family": "low_overlap_variant",
            "profiles": ("exploration",),
            "timeout_sec": 900,
            "variant_suffix": "low_overlap",
            "shared_execution_group_id": "local_registered__htdemucs_ft__low_overlap",
            "unsupported_parameters": ("overlap",),
        },
    )


def build_candidate_job_registry(
    profile: str | None = None,
    *,
    registry: ModelRegistry | None = None,
) -> list[CandidateJobConfig]:
    if profile is not None and profile not in ALLOWED_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")
    registry = registry or default_model_registry()
    entrypoint_exists = local_registered_entrypoint_exists()
    configs: list[CandidateJobConfig] = []
    for spec in config_specs():
        profiles = tuple(spec["profiles"])
        if profile is not None and profile not in profiles:
            continue
        model_available, model_notes = model_can_run(registry, str(spec["model_id"]))
        can_run = entrypoint_exists and model_available
        unavailable_notes: list[str] = []
        if not entrypoint_exists:
            unavailable_notes.append(f"missing_entrypoint:{REGISTERED_MODEL_ENTRYPOINT}")
        unavailable_notes.extend(model_notes)
        unsupported_parameters = tuple(str(value) for value in spec.get("unsupported_parameters", ()))
        unavailable_notes.extend(f"unsupported_parameter:{value}" for value in unsupported_parameters)
        model_id = str(spec["model_id"])
        target_mode = str(spec["target_mode"])
        variant_suffix = str(spec.get("variant_suffix") or "").strip()
        config_id = f"local_registered__{model_id}__{target_mode}"
        if variant_suffix:
            config_id = f"{config_id}__{variant_suffix}"
        produces_vocal = bool(spec.get("produces_vocal", target_mode in {"vocal", "both"}))
        produces_minus = bool(spec.get("produces_minus", target_mode in {"minus", "both"}))
        materialized_modes = tuple(str(value) for value in spec.get("materialized_candidate_modes", (target_mode,)))
        can_run = can_run and not unsupported_parameters
        configs.append(
            CandidateJobConfig(
                candidate_config_id=config_id,
                candidate_family=str(spec["family"]),
                target_mode=target_mode,
                source_model_id=model_id,
                source_tool=REGISTERED_MODEL_ENTRYPOINT,
                entrypoint_type="python_script" if can_run else "unavailable",
                command_template=registered_model_command_template(),
                expected_outputs={"vocal_path": "", "minus_path": ""},
                gpu_required=True,
                timeout_sec=int(spec["timeout_sec"]),
                max_retries=0,
                profile_allowed=profiles,
                safety_notes=tuple(
                    [
                        "uses_existing_model_registry_entry",
                        "no_model_download",
                        "no_training",
                        "no_cleanup_chain",
                    ]
                    + unavailable_notes
                ),
                supported_parameters=registered_model_supported_parameters(),
                shared_execution_group_id=str(spec.get("shared_execution_group_id") or config_id),
                produces_vocal=produces_vocal,
                produces_minus=produces_minus,
                produces_both=bool(produces_vocal and produces_minus),
                materialized_candidate_modes=materialized_modes,
                reuse_outputs_across_modes=bool(spec.get("reuse_outputs_across_modes", False)),
                target_stem=str(spec["target_stem"]),
            )
        )
    return sorted(configs, key=lambda config: config.candidate_config_id)


def get_candidate_job_config(
    candidate_config_id: str,
    *,
    registry_configs: Iterable[CandidateJobConfig] | None = None,
    profile: str | None = None,
) -> CandidateJobConfig:
    configs = list(registry_configs) if registry_configs is not None else build_candidate_job_registry(profile=profile)
    for config in configs:
        if config.candidate_config_id == candidate_config_id:
            return config
    raise KeyError(f"Unknown candidate_config_id: {candidate_config_id}")
