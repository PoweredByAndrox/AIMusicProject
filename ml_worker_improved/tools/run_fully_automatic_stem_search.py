"""Run a fail-closed fully automatic six-stem search loop."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import soundfile as sf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.score_stem_contamination import score_stem_contamination
from tools.build_training_first_root_cause_queue import build_root_cause_queue
from tools.stem_forensic_crosscheck import (
    score_cross_stem_forensic_matrix,
    score_pair_forensic,
    validate_listen_pack_forensic,
)
from tools.stem_region_repair_common import (
    ISSUE_BAD_RECONSTRUCTION,
    ISSUE_INSTRUMENT_LEAK,
    ISSUE_STEREO_COLLAPSE,
    make_region_mask,
)
from tools.stem_route_policy_common import (
    ACTION_BLOCK_UNKNOWN,
    ACTION_PRIORITIZE_ROUTE,
    ACTION_QUARANTINE_ROUTE,
    ACTIONS as ROUTE_POLICY_ACTIONS,
    extract_lane_routes,
    extract_route_signature,
    normalize_policy_action,
)
from tools.stem_route_policy_model import load_stem_route_policy_checkpoint, score_policy_row
from tools.persistent_separator_backend import (
    MODE_PERSISTENT_SERVER,
    MODE_SUBPROCESS,
    PersistentSeparatorClient,
    PersistentSeparatorRejected,
    PersistentSeparatorUnavailable,
    SeparatorBackendConfig,
    make_separator_backend_config,
    separator_cache_key,
)
from tools.stem_contamination_common import (
    FORBIDDEN_SOURCES,
    NON_VOCAL_STEMS,
    STEMS,
    VERDICTS,
    missing_required_stems,
    validate_no_artifact_rag_output,
)
from tools.stem_contamination_model import band_limited_audio
from tools.build_vocal_depercussive_teacher_target import _repair_regions as _repair_vocal_depercussive_regions
from worker.model_registry import default_model_registry


CommandRunner = Callable[[list[str], int], None]
DEFAULT_CONTAMINATION_MODEL_SEARCH_ROOTS = (
    Path("/project_artifacts/stem_eval"),
    Path("/host_artifacts/stem_eval"),
    Path("/repo/artifacts/stem_eval"),
    Path("artifacts/stem_eval"),
    Path("../artifacts/stem_eval"),
)
TARGET_SPECIFIC_ROLE_OUTPUTS = {
    "six_stems": tuple(STEMS),
    "vocals": ("vocals",),
    "minus": ("minus",),
    "drums_support": ("drums",),
    "bass_support": ("bass",),
    "guitar_support": ("guitar",),
    "piano_support": ("piano",),
    "other_support": ("other",),
    "non_vocal_support": ("drums", "bass", "other"),
    "four_stem_non_vocal": ("drums", "bass", "other"),
    "non_vocal_stems": tuple(NON_VOCAL_STEMS),
}
MODEL_SPLIT_V2_PROFILE = "model_split_v2"
PERSISTENT_MPS_MINUS_PAIR_PROFILE = "persistent_mps_minus_pair"
PERSISTENT_MPS_MINUS_PAIR_ROUTE_ID = "becruily_instrumental_vocals"
MODEL_SPLIT_V2_DELIVERY_PAIR_ROLES = {"vocals", "minus"}
DEBLEED_TARGET_STEMS = tuple(dict.fromkeys((*NON_VOCAL_STEMS, "minus")))
MODEL_SPLIT_V2_FAIL_FAST_MUTATION_KEYS = (
    "minus_vocal_debleed",
    "vocal_projection_debleed",
    "conservation_repair_stem",
    "stem_merges",
    "post_minus_vocal_debleed",
    "model_split_composite_repair_options",
)
ROUTE_TIMEOUT_OVERRIDES_SEC = {
    "mdx23c_vocals": 45,
    "mdx23c_vocals:instrumental": 45,
}
ROUTE_BUDGET_PROFILE_FULL = "full"
ROUTE_BUDGET_PROFILE_FAST = "fast"
ROUTE_BUDGET_PROFILES = (ROUTE_BUDGET_PROFILE_FULL, ROUTE_BUDGET_PROFILE_FAST)
FAST_ROUTE_TIMEOUT_OVERRIDES_SEC = {
    "mdx23c_vocals": 30,
    "mdx23c_vocals:instrumental": 30,
    "becruily_instrumental_vocals": 120,
    "becruily_instrumental_vocals:instrumental": 120,
    "melband_roformer_vocals": 30,
    "melband_roformer_vocals:instrumental": 30,
    "scnet_xl_ihf_musdb18:bass": 30,
    "scnet_xl_ihf_musdb18:drums_bass_other": 30,
}
DEFAULT_VOCAL_ROUTE_ORDER = (
    "audio_separator_bs_roformer_vocals",
    "mdx23c_vocals",
    "melband_roformer_vocals",
    "becruily_instrumental_vocals",
)
MODEL_SPLIT_V2_VOCAL_LANE = (
    "bs_mamba2_vocals",
    "audio_separator_bs_roformer_vocals",
    "mdx23c_vocals",
    "melband_roformer_vocals",
)
MODEL_SPLIT_V2_MINUS_LANE = (
    "bs_mamba2_vocals:instrumental",
    "becruily_instrumental_vocals",
    "audio_separator_bs_roformer_vocals:instrumental",
    "mdx23c_vocals:instrumental",
    "htdemucs_ft:instrumental",
)
DEFAULT_MIN_GPU_FREE_MIB = 4096
MODEL_SPLIT_V2_STEM_LANE = (
    "htdemucs_6s",
    "hybrid_htdemucs_ft_bs_roformer_6s",
    "hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first",
)
MODEL_SPLIT_V2_BASS_SUPPORT_LANE = (
    "scnet_xl_ihf_musdb18:bass",
)
MODEL_SPLIT_V2_NON_VOCAL_SUPPORT_LANE = (
    "scnet_xl_ihf_musdb18:drums_bass_other",
)
TEACHER_MASK_CHECKPOINT_ROLES = (
    "vocals",
    "minus",
    "drums_support",
    "bass_support",
    "guitar_support",
    "piano_support",
    "other_support",
)
MODEL_SPLIT_V2_REGISTERED_ROUTE_BASES = {
    "bs_mamba2_vocals",
    "htdemucs_ft",
    "hybrid_htdemucs_ft_bs_roformer_6s",
    "hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first",
    "scnet_xl_ihf_musdb18",
}
DEFAULT_VOCAL_ROUTE_REQUIRED_PATHS = {
    "audio_separator_bs_roformer_vocals": (
        "/app/tools/export_audio_separator_vocal_full.py",
        "/opt/audio-separator-venv/bin/python",
        "/app/models/audio_separator/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "/app/models/audio_separator/model_bs_roformer_ep_317_sdr_12.9755.yaml",
    ),
    "mdx23c_vocals": (
        "/app/tools/export_mdx23c_vocal_full.py",
        "/app/vendor/mdx23c_vocals_official/config_vocals_mdx23c.yaml",
        "/app/vendor/mdx23c_vocals_official/model_vocals_mdx23c_sdr_10.17.ckpt",
    ),
    "melband_roformer_vocals": (
        "/app/tools/export_melband_vocal_full.py",
        "/app/vendor/mel_band_roformer_vocal_model",
        "/app/vendor/mel_band_roformer_vocal_model_weights/MelBandRoformer.ckpt",
    ),
    "becruily_instrumental_vocals": (
        "/app/tools/export_melband_vocal_full.py",
        "/app/vendor/mel_band_roformer_vocal_model",
        "/app/vendor/becruily_instrumental/config_instrumental_becruily.yaml",
        "/app/vendor/becruily_instrumental/mel_band_roformer_instrumental_becruily.ckpt",
    ),
}
MIN_TRAILING_WINDOW_SEC = 1.0


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _remove_listen_pack(output_dir: Path) -> None:
    shutil.rmtree(output_dir / "listen", ignore_errors=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if _parse_bool(os.environ.get("STEM_SEARCH_ECHO_PROGRESS", "0")):
        print(line, file=sys.stderr, flush=True)


def _collect_cuda_preflight(*, require_gpu: bool, min_free_mib: int) -> dict[str, Any]:
    checked_min_free_mib = int(max(0, min_free_mib))
    preflight: dict[str, Any] = {
        "schema": "stem_search_cuda_preflight.v1",
        "require_gpu": bool(require_gpu),
        "min_free_mib": checked_min_free_mib,
        "status": "not_required",
        "reason": "",
    }
    if not require_gpu:
        return preflight
    if checked_min_free_mib <= 0:
        preflight.update({"status": "not_checked", "reason": "min_free_mib_not_set"})
        return preflight
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total,utilization.gpu,name",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        preflight.update({"status": "blocked", "reason": "nvidia_smi_unavailable", "error": str(exc)})
        return preflight
    if completed.returncode != 0:
        preflight.update(
            {
                "status": "blocked",
                "reason": "nvidia_smi_failed",
                "stderr": completed.stderr.strip()[:500],
            }
        )
        return preflight
    first_row = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
    fields = [field.strip() for field in first_row.split(",", 3)]
    if len(fields) < 4:
        preflight.update({"status": "blocked", "reason": "nvidia_smi_unparseable", "stdout": first_row[:500]})
        return preflight
    free_mib = int(float(fields[0]))
    total_mib = int(float(fields[1]))
    utilization_pct = int(float(fields[2]))
    preflight.update(
        {
            "free_mib": free_mib,
            "total_mib": total_mib,
            "used_mib": max(0, total_mib - free_mib),
            "utilization_pct": utilization_pct,
            "device_name": fields[3],
        }
    )
    if free_mib < checked_min_free_mib:
        preflight.update({"status": "blocked", "reason": "gpu_free_memory_too_low"})
    else:
        preflight.update({"status": "passed", "reason": ""})
    return preflight


def _collect_separator_accelerator_preflight(
    *,
    require_gpu: bool,
    min_free_mib: int,
    separator_backend_config: SeparatorBackendConfig,
) -> dict[str, Any]:
    device = str(separator_backend_config.device or "").strip().lower()
    if require_gpu and device == "mps" and separator_backend_config.mode == MODE_PERSISTENT_SERVER:
        return {
            "schema": "stem_search_accelerator_preflight.v1",
            "require_gpu": True,
            "accelerator": "mps",
            "status": "delegated_to_persistent_backend",
            "reason": "mps_backend_preflight_delegated",
            "server_url": separator_backend_config.server_url,
        }
    return _collect_cuda_preflight(require_gpu=require_gpu, min_free_mib=min_free_mib)


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate)
    return path


def _match_audio_shape(
    audio: np.ndarray,
    source_rate: int,
    *,
    target_sample_rate: int | None,
    target_sample_count: int | None,
    target_channel_count: int | None = None,
) -> tuple[np.ndarray, int]:
    output = np.asarray(audio, dtype=np.float32)
    if output.ndim == 1:
        output = output[:, None]
    output_rate = int(source_rate)
    if target_sample_rate and int(target_sample_rate) != output_rate and len(output) > 0:
        old_x = np.linspace(0.0, 1.0, num=len(output), endpoint=False)
        new_len = max(1, int(round(len(output) * float(target_sample_rate) / float(output_rate))))
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        output = np.stack(
            [
                np.interp(new_x, old_x, output[:, channel]).astype(np.float32)
                for channel in range(output.shape[1])
            ],
            axis=1,
        )
        output_rate = int(target_sample_rate)
    if target_sample_count is not None:
        desired = int(target_sample_count)
        if desired < 0:
            raise ValueError("target_sample_count must be non-negative")
        if len(output) > desired:
            output = output[:desired]
        elif len(output) < desired:
            pad = np.zeros((desired - len(output), output.shape[1]), dtype=np.float32)
            output = np.concatenate([output, pad], axis=0)
    if target_channel_count is not None and int(target_channel_count) > 0 and output.shape[1] != int(target_channel_count):
        desired_channels = int(target_channel_count)
        if output.shape[1] == 1:
            output = np.repeat(output, desired_channels, axis=1)
        elif desired_channels == 1:
            output = np.mean(output, axis=1, keepdims=True).astype(np.float32)
        elif output.shape[1] > desired_channels:
            output = output[:, :desired_channels]
        else:
            pad = np.zeros((output.shape[0], desired_channels - output.shape[1]), dtype=np.float32)
            output = np.concatenate([output, pad], axis=1)
    return output.astype(np.float32), output_rate


def _copy_audio_to_match_shape(
    source_path: Path,
    output_path: Path,
    *,
    target_sample_rate: int | None,
    target_sample_count: int | None,
    target_channel_count: int | None = None,
) -> Path:
    audio, sample_rate = sf.read(source_path, dtype="float32", always_2d=True)
    audio, sample_rate = _match_audio_shape(
        audio,
        int(sample_rate),
        target_sample_rate=target_sample_rate,
        target_sample_count=target_sample_count,
        target_channel_count=target_channel_count,
    )
    return _write_audio(output_path, audio, sample_rate)


def _render_template(value: str, replacements: Mapping[str, str]) -> str:
    rendered = value
    for key, replacement in replacements.items():
        rendered = rendered.replace("{" + key + "}", replacement)
    return rendered


def render_recipe_command(command_template: list[Any], replacements: Mapping[str, str]) -> list[str]:
    return [_render_template(str(part), replacements) for part in command_template]


def default_command_runner(command: list[str], timeout_sec: int) -> None:
    subprocess.run(command, check=True, timeout=timeout_sec)


def _default_vocal_route_available(route_id: str) -> bool:
    return all(Path(path).exists() for path in DEFAULT_VOCAL_ROUTE_REQUIRED_PATHS.get(route_id, ()))


def _available_default_vocal_routes(available_vocal_routes: tuple[str, ...] | None) -> tuple[str, ...]:
    if available_vocal_routes is not None:
        return tuple(route for route in available_vocal_routes if route in DEFAULT_VOCAL_ROUTE_ORDER)
    return tuple(route for route in DEFAULT_VOCAL_ROUTE_ORDER if _default_vocal_route_available(route))


def _contamination_manifest_sort_key(path: Path) -> tuple[int, float, str]:
    cycle_name = path.parent.parent.name
    match = re.search(r"(\d+)$", cycle_name)
    cycle_number = int(match.group(1)) if match else -1
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return cycle_number, mtime, str(path)


def _discover_default_contamination_model_manifest(
    search_roots: tuple[Path, ...] = DEFAULT_CONTAMINATION_MODEL_SEARCH_ROOTS,
) -> Path | None:
    env_path = os.environ.get("STEM_CONTAMINATION_MODEL_MANIFEST", "").strip()
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return path

    candidates: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        candidates.extend(path for path in root.glob("cycle*/model/manifest.json") if path.is_file())
    if not candidates:
        return None
    return sorted(candidates, key=_contamination_manifest_sort_key)[-1]


def _demucs_six_stem_command_template(
    *,
    segment: str,
    overlap: str,
    shifts: str,
    residual_stem: str,
) -> list[str]:
    command_template = [
        "python",
        "tools/run_demucs_six_stem_candidate.py",
        "--input-audio",
        "{input_audio}",
        "--stems-dir",
        "{stems_dir}",
        "--work-dir",
        "{output_dir}/work",
        "--manifest",
        "{output_dir}/demucs.json",
        "--model-name",
        "htdemucs_6s",
        "--segment-sec",
        segment,
        "--overlap",
        overlap,
        "--shifts",
        shifts,
        "--device",
        "cuda",
    ]
    if residual_stem:
        command_template.extend(["--residual-correction-stem", residual_stem])
    command_template.extend(["--timeout-sec", "900"])
    return command_template


def _vocal_route_command_template(route_id: str) -> list[str]:
    if route_id == "bs_mamba2_vocals":
        return _registered_model_command_template(route_id, target_stem="vocals")
    if route_id == "audio_separator_bs_roformer_vocals":
        return [
            "python",
            "tools/export_audio_separator_vocal_full.py",
            "{input_audio}",
            "{stems_dir}",
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
        ]
    if route_id == "mdx23c_vocals":
        return [
            "python",
            "tools/export_mdx23c_vocal_full.py",
            "{input_audio}",
            "{stems_dir}",
        ]
    if route_id == "melband_roformer_vocals":
        return [
            "python",
            "tools/export_melband_vocal_full.py",
            "{input_audio}",
            "{stems_dir}",
        ]
    if route_id == "becruily_instrumental_vocals":
        return [
            "python",
            "tools/export_melband_vocal_full.py",
            "{input_audio}",
            "{stems_dir}",
            "--model-path",
            "/app/vendor/becruily_instrumental/mel_band_roformer_instrumental_becruily.ckpt",
            "--config-path",
            "/app/vendor/becruily_instrumental/config_instrumental_becruily.yaml",
            "--manifest-model-id",
            "becruily_instrumental_vocals",
            "--manifest-model-display-name",
            "Becruily MelBand RoFormer instrumental checkpoint",
        ]
    raise ValueError(f"unsupported default vocal route: {route_id}")


def _registered_six_stem_command_template(model_id: str) -> list[str]:
    return [
        "python",
        "tools/run_registered_six_stem_candidate.py",
        "--input-audio",
        "{input_audio}",
        "--stems-dir",
        "{stems_dir}",
        "--work-dir",
        "{output_dir}/work",
        "--manifest",
        "{output_dir}/registered.json",
        "--model-id",
        model_id,
        "--device",
        "cuda",
        "--timeout-sec",
        "900",
    ]


def _stem_lane_command_template(model_id: str) -> list[str]:
    if model_id == "htdemucs_6s":
        return _demucs_six_stem_command_template(segment="6", overlap="0.25", shifts="1", residual_stem="")
    if model_id in {
        "hybrid_htdemucs_ft_bs_roformer_6s",
        "hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first",
    }:
        return _registered_six_stem_command_template(model_id)
    raise ValueError(f"unsupported model-split stem lane model: {model_id}")


def _registered_model_command_template(model_id: str, *, target_stem: str | None = None) -> list[str]:
    command = [
        "python",
        "tools/run_registered_model_candidate.py",
        "{input_audio}",
        "{stems_dir}",
        "--model-id",
        model_id,
        "--device",
        "cuda",
    ]
    if target_stem:
        command.extend(["--target-stem", target_stem])
    return command


def _teacher_mask_checkpoint_command_template(*, checkpoint: str, stem: str) -> list[str]:
    return [
        "python",
        "tools/apply_teacher_mask_checkpoint_candidate.py",
        "{input_audio}",
        "{stems_dir}",
        "--checkpoint",
        checkpoint,
        "--stem",
        stem,
        "--device",
        "cuda",
    ]


def _registered_route_is_available(route_id: str) -> bool:
    base_route = str(route_id or "").split(":", 1)[0]
    if base_route not in MODEL_SPLIT_V2_REGISTERED_ROUTE_BASES:
        return True
    try:
        registry = default_model_registry()
        entry = registry.get(base_route)
        return registry.availability(entry).status == "available"
    except Exception:
        return False


def _filter_available_model_split_routes(routes: Sequence[str]) -> tuple[str, ...]:
    return tuple(route for route in routes if _registered_route_is_available(route))


def _prefers_resident_separator(route_id: str) -> bool:
    return str(route_id or "").split(":", 1)[0] in {
        "audio_separator_bs_roformer_vocals",
        "becruily_instrumental_vocals",
        "melband_roformer_vocals",
    }


def _minus_lane_command_template(route_id: str) -> list[str]:
    base_route = route_id.split(":", 1)[0]
    if base_route == "bs_mamba2_vocals":
        return _registered_model_command_template(base_route, target_stem="minus")
    if base_route in {
        "audio_separator_bs_roformer_vocals",
        "mdx23c_vocals",
        "melband_roformer_vocals",
        "becruily_instrumental_vocals",
    }:
        return _vocal_route_command_template(base_route)
    if base_route == "htdemucs_ft":
        return _registered_model_command_template(base_route, target_stem="minus")
    raise ValueError(f"unsupported model-split minus lane model: {route_id}")


def _bass_support_lane_command_template(route_id: str) -> list[str]:
    base_route, _, target_stem = route_id.partition(":")
    return _registered_model_command_template(base_route, target_stem=target_stem or "bass")


def _non_vocal_support_lane_command_template(route_id: str) -> list[str]:
    base_route = route_id.split(":", 1)[0]
    return [
        "python",
        "tools/run_registered_non_vocal_candidate.py",
        "--input-audio",
        "{input_audio}",
        "--stems-dir",
        "{stems_dir}",
        "--work-dir",
        "{output_dir}/work",
        "--manifest",
        "{output_dir}/registered_non_vocal.json",
        "--model-id",
        base_route,
        "--device",
        "cuda",
        "--timeout-sec",
        "900",
    ]


def _model_split_stem_region_repair_options(
    stem_region_repair_checkpoints: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    if not stem_region_repair_checkpoints:
        return options

    def repair_config(*, raw_checkpoint: Mapping[str, Any], stem: str, issue_type: str, checkpoint: str) -> dict[str, Any]:
        config: dict[str, Any] = {"stem": stem, "issue_type": issue_type, "checkpoint": checkpoint}
        if stem == "minus" and issue_type == "vocal_leak":
            config["mix_preserve_companion_stem"] = "vocals"
        for key in (
            "forensic_source_condition_stem",
            "mix_preserve_companion_stem",
            "delivery_pair_compensation_strength",
            "forensic_high_band_hz",
        ):
            value = raw_checkpoint.get(key)
            if value not in (None, ""):
                config[key] = value
        region = raw_checkpoint.get("region")
        if isinstance(region, Mapping):
            config["region"] = {
                "start_sec": float(region.get("start_sec", 0.0)),
                "end_sec": float(region.get("end_sec", 0.0)),
            }
        return config

    sequence_configs: list[dict[str, str]] = []
    for index, raw_checkpoint in enumerate(stem_region_repair_checkpoints, start=1):
        if not isinstance(raw_checkpoint, Mapping):
            continue
        checkpoint = str(raw_checkpoint.get("checkpoint") or raw_checkpoint.get("checkpoint_path") or "").strip()
        if not checkpoint:
            continue
        stem = str(raw_checkpoint.get("stem") or "bass").strip().lower()
        issue_type = str(raw_checkpoint.get("issue_type") or ISSUE_STEREO_COLLAPSE).strip().lower()
        repair_id = _safe_artifact_segment(
            str(raw_checkpoint.get("repair_id") or f"sr{index:03d}"),
            fallback=f"sr{index:03d}",
        )
        config = repair_config(raw_checkpoint=raw_checkpoint, stem=stem, issue_type=issue_type, checkpoint=checkpoint)
        sequence_configs.append(dict(config))
        options.append(
            {
                "repair_id": repair_id,
                "type": "stem_region_repair",
                **config,
            }
        )
        options.append(
            {
                "repair_id": f"{repair_id}_vp100",
                "type": "stem_region_repair+vocal_projection_debleed",
                "stem_region_repair": dict(config),
                "vocal_projection_debleed": {
                    "strength": 1.0,
                    "target_stems": list(NON_VOCAL_STEMS),
                },
            }
        )
        options.insert(
            len(options) - 1,
            {
                "repair_id": f"{repair_id}_vp100_pr16",
                "type": "stem_region_repair+vocal_projection_debleed+delivery_pair_residual",
                "stem_region_repair": dict(config),
                "vocal_projection_debleed": {
                    "strength": 1.0,
                    "target_stems": list(NON_VOCAL_STEMS),
                },
                "delivery_pair_residual": {
                    "target_stem": "minus",
                    "max_residual_ratio": 0.16,
                },
            },
        )
    if len(sequence_configs) > 1:
        options.insert(
            0,
            {
                "repair_id": "srseq",
                "type": "stem_region_repair",
                "stem_region_repairs": sequence_configs,
            },
        )
        options.insert(
            1,
            {
                "repair_id": "srseq_bs400",
                "type": "stem_region_repair+bass_stereo_width",
                "stem_region_repairs": sequence_configs,
                "bass_stereo_width": {
                    "side_gain": 4.0,
                },
            },
        )
        options.insert(
            2,
            {
                "repair_id": "srseq_vp100",
                "type": "stem_region_repair+vocal_projection_debleed",
                "stem_region_repairs": sequence_configs,
                "vocal_projection_debleed": {
                    "strength": 1.0,
                    "target_stems": list(NON_VOCAL_STEMS),
                },
            },
        )
        options.insert(
            3,
            {
                "repair_id": "srseq_vp100_bs400",
                "type": "stem_region_repair+vocal_projection_debleed+bass_stereo_width",
                "stem_region_repairs": sequence_configs,
                "vocal_projection_debleed": {
                    "strength": 1.0,
                    "target_stems": list(NON_VOCAL_STEMS),
                },
                "bass_stereo_width": {
                    "side_gain": 4.0,
                },
            },
        )
        options.insert(
            4,
            {
                "repair_id": "srseq_vp100_pr16",
                "type": "stem_region_repair+vocal_projection_debleed+delivery_pair_residual",
                "stem_region_repairs": sequence_configs,
                "vocal_projection_debleed": {
                    "strength": 1.0,
                    "target_stems": list(NON_VOCAL_STEMS),
                },
                "delivery_pair_residual": {
                    "target_stem": "minus",
                    "max_residual_ratio": 0.16,
                },
            },
        )
    return options


def _model_split_teacher_mask_checkpoint_routes(
    checkpoints: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoints or [], start=1):
        role = str(checkpoint.get("role") or "").strip()
        stem = str(checkpoint.get("stem") or "").strip()
        checkpoint_path = str(checkpoint.get("checkpoint") or "").strip()
        if role not in TEACHER_MASK_CHECKPOINT_ROLES:
            raise ValueError(f"unsupported teacher-mask checkpoint role: {role}")
        if not stem or not checkpoint_path:
            raise ValueError("teacher-mask checkpoint route requires stem and checkpoint")
        expected_outputs = TARGET_SPECIFIC_ROLE_OUTPUTS.get(role, ())
        if stem not in expected_outputs:
            raise ValueError(f"teacher-mask checkpoint stem {stem} does not match role {role}")
        route_id = _safe_artifact_segment(
            str(checkpoint.get("route_id") or f"teacher_mask_{role}_{index:03d}"),
            fallback=f"tm{index:03d}",
        )
        routes.append(
            {
                "role": role,
                "lane": "teacher_mask_checkpoint_lane",
                "route_id": route_id,
                "teacher_mask_checkpoint_path": checkpoint_path,
                "command_template": _teacher_mask_checkpoint_command_template(
                    checkpoint=checkpoint_path,
                    stem=stem,
                ),
            }
        )
    return routes


def build_model_split_v2_recipe(
    *,
    demucs_available: bool | None = None,
    available_vocal_routes: tuple[str, ...] | None = None,
    contamination_model_manifest: Path | str | None = None,
    stem_region_repair_checkpoints: Sequence[Mapping[str, Any]] | None = None,
    teacher_mask_checkpoints: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if demucs_available is None:
        demucs_available = shutil.which("demucs") is not None
    if not demucs_available:
        return {
            "schema": "stem_search_recipe.v2",
            "profile": MODEL_SPLIT_V2_PROFILE,
            "recipe_source": "model_split_v2",
            "available": False,
            "unavailable_reason": "missing_demucs_cli",
            "candidates": [],
        }

    model_split_vocal_lane = _filter_available_model_split_routes(MODEL_SPLIT_V2_VOCAL_LANE)
    model_split_minus_lane = _filter_available_model_split_routes(MODEL_SPLIT_V2_MINUS_LANE)
    model_split_stem_lane = _filter_available_model_split_routes(MODEL_SPLIT_V2_STEM_LANE)
    model_split_bass_support_lane = _filter_available_model_split_routes(MODEL_SPLIT_V2_BASS_SUPPORT_LANE)
    model_split_non_vocal_support_lane = _filter_available_model_split_routes(MODEL_SPLIT_V2_NON_VOCAL_SUPPORT_LANE)
    if not model_split_vocal_lane or not model_split_minus_lane or not model_split_stem_lane:
        return {
            "schema": "stem_search_recipe.v2",
            "profile": MODEL_SPLIT_V2_PROFILE,
            "recipe_source": "model_split_v2",
            "available": False,
            "unavailable_reason": "missing_available_model_split_lane",
            "candidates": [],
            "target_specific_routes": {
                "vocal_lane": list(model_split_vocal_lane),
                "minus_lane": list(model_split_minus_lane),
                "stem_lane": list(model_split_stem_lane),
                "bass_support_lane": list(model_split_bass_support_lane),
                "non_vocal_support_lane": list(model_split_non_vocal_support_lane),
            },
        }

    vocal_lane = tuple(
        route for route in MODEL_SPLIT_V2_VOCAL_LANE if available_vocal_routes is None or route in set(available_vocal_routes)
    )
    vocal_lane = tuple(route for route in vocal_lane if route in set(model_split_vocal_lane))
    if available_vocal_routes is not None:
        # Keep the contract-visible lane list stable even when callers restrict
        # the first executable routes in tests or diagnostics.
        vocal_lane_manifest = tuple(model_split_vocal_lane)
    else:
        vocal_lane_manifest = tuple(model_split_vocal_lane)
    executable_vocal_lane = vocal_lane or tuple(model_split_vocal_lane)
    learned_repair_options = _model_split_stem_region_repair_options(stem_region_repair_checkpoints)
    teacher_mask_routes = _model_split_teacher_mask_checkpoint_routes(teacher_mask_checkpoints)
    candidates: list[dict[str, Any]] = []
    for index in range(10):
        stem_route = model_split_stem_lane[index % len(model_split_stem_lane)]
        vocal_route = executable_vocal_lane[index % len(executable_vocal_lane)]
        minus_route = model_split_minus_lane[index % len(model_split_minus_lane)]
        minus_output_map = {"minus": "minus" if minus_route.startswith("htdemucs_ft") else "instrumental"}
        candidate_id = f"c{index + 1:03d}"
        target_specific_commands = [
            {
                "role": "six_stems",
                "lane": "stem_lane",
                "route_id": stem_route,
                "command_template": _stem_lane_command_template(stem_route),
            },
            {
                "role": "vocals",
                "lane": "vocal_lane",
                "route_id": vocal_route,
                "prefer_resident": _prefers_resident_separator(vocal_route),
                "command_template": _vocal_route_command_template(vocal_route),
            },
            {
                "role": "minus",
                "lane": "minus_lane",
                "route_id": minus_route,
                "output_map": minus_output_map,
                "prefer_resident": _prefers_resident_separator(minus_route),
                "command_template": _minus_lane_command_template(minus_route),
            },
        ]
        if index == 1:
            if model_split_bass_support_lane:
                bass_support_route = model_split_bass_support_lane[0]
                target_specific_commands.append(
                    {
                        "role": "bass_support",
                        "lane": "bass_support_lane",
                        "route_id": bass_support_route,
                        "command_template": _bass_support_lane_command_template(bass_support_route),
                    }
                )
        if index in {0, 2}:
            if model_split_non_vocal_support_lane:
                support_route = model_split_non_vocal_support_lane[0]
                target_specific_commands.append(
                    {
                        "role": "non_vocal_support",
                        "lane": "non_vocal_support_lane",
                        "route_id": support_route,
                        "command_template": _non_vocal_support_lane_command_template(support_route),
                    }
                )
        if teacher_mask_routes:
            target_specific_commands.extend(teacher_mask_routes)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "profile": MODEL_SPLIT_V2_PROFILE,
                "route_family": (
                    f"model_split_v2_{stem_route}_{vocal_route}_{minus_route.replace(':', '_')}"
                    + ("_teacher_mask_checkpoint" if teacher_mask_routes else "")
                ),
                "required_stems": list(STEMS),
                "minus_source": "minus_lane",
                "minus_rebuilt_from_non_vocal_stems": False,
                "model_split_lanes": {
                    "vocals": "vocal_lane",
                    "minus": "minus_lane",
                    "stems": "stem_lane",
                    "bass_support": "bass_support_lane",
                    "non_vocal_support": "non_vocal_support_lane",
                },
                "damage_gates": {
                    "enabled": True,
                    "min_non_vocal_rms": 1e-5,
                    "min_relative_stereo_width": 0.20,
                    "min_mix_stereo_width": 0.02,
                },
                "voice_presence_gate": {
                    "enabled": True,
                    "max_projection_ratio": 0.12,
                    "max_projection_coeff": 0.18,
                },
                "delivery_pair_reconstruction": {
                    "enabled": True,
                    "max_delivery_pair_residual_ratio": 0.06,
                    "six_stem_diagnostic_residual_ratio": 0.01,
                },
                "model_split_composite_repair_options": [
                    {"repair_id": "none", "type": "none"},
                    *learned_repair_options,
                    {
                        "repair_id": "vp100",
                        "type": "vocal_projection_debleed",
                        "strength": 1.0,
                        "target_stems": list(NON_VOCAL_STEMS),
                    },
                    {
                        "repair_id": "bs250",
                        "type": "bass_stereo_width",
                        "side_gain": 2.5,
                    },
                    {
                        "repair_id": "bs400",
                        "type": "bass_stereo_width",
                        "side_gain": 4.0,
                    },
                    {
                        "repair_id": "pg08",
                        "type": "delivery_pair_gain",
                        "target_stem": "minus",
                        "max_abs_gain_delta": 0.08,
                    },
                    {
                        "repair_id": "pr16",
                        "type": "delivery_pair_residual",
                        "target_stem": "minus",
                        "max_residual_ratio": 0.16,
                    },
                    {
                        "repair_id": "bs400_pr16",
                        "type": "bass_stereo_width+delivery_pair_residual",
                        "bass_stereo_width": {
                            "side_gain": 4.0,
                        },
                        "delivery_pair_residual": {
                            "target_stem": "minus",
                            "max_residual_ratio": 0.16,
                        },
                    },
                    {
                        "repair_id": "vp100_bs400",
                        "type": "vocal_projection_debleed+bass_stereo_width",
                        "vocal_projection_debleed": {
                            "strength": 1.0,
                            "target_stems": list(NON_VOCAL_STEMS),
                        },
                        "bass_stereo_width": {
                            "side_gain": 4.0,
                        },
                    },
                    {
                        "repair_id": "bs400_pg08",
                        "type": "bass_stereo_width+delivery_pair_gain",
                        "bass_stereo_width": {
                            "side_gain": 4.0,
                        },
                        "delivery_pair_gain": {
                            "target_stem": "minus",
                            "max_abs_gain_delta": 0.08,
                        },
                    },
                    {
                        "repair_id": "vp100_pg08",
                        "type": "vocal_projection_debleed+delivery_pair_gain",
                        "vocal_projection_debleed": {
                            "strength": 1.0,
                            "target_stems": list(NON_VOCAL_STEMS),
                        },
                        "delivery_pair_gain": {
                            "target_stem": "minus",
                            "max_abs_gain_delta": 0.08,
                        },
                    },
                    {
                        "repair_id": "vp100_bs400_pg08",
                        "type": "vocal_projection_debleed+bass_stereo_width+delivery_pair_gain",
                        "vocal_projection_debleed": {
                            "strength": 1.0,
                            "target_stems": list(NON_VOCAL_STEMS),
                        },
                        "bass_stereo_width": {
                            "side_gain": 4.0,
                        },
                        "delivery_pair_gain": {
                            "target_stem": "minus",
                            "max_abs_gain_delta": 0.08,
                        },
                    },
                ],
                "target_specific_commands": target_specific_commands,
            }
        )

    recipe: dict[str, Any] = {
        "schema": "stem_search_recipe.v2",
        "profile": MODEL_SPLIT_V2_PROFILE,
        "recipe_source": "model_split_v2",
        "available": True,
        "target_specific_routes": {
            "vocal_lane": list(vocal_lane_manifest),
            "minus_lane": list(model_split_minus_lane),
            "stem_lane": list(model_split_stem_lane),
            "bass_support_lane": list(model_split_bass_support_lane),
            "non_vocal_support_lane": list(model_split_non_vocal_support_lane),
            "teacher_mask_checkpoint_lane": [
                str(route.get("route_id") or "") for route in teacher_mask_routes
            ],
            "auxiliary_non_vocal_evidence": ["scnet_xl_ihf_musdb18"],
        },
        "model_split_composite_max_candidates": 864,
        "timeout_sec": 900,
        "candidates": candidates,
    }
    if contamination_model_manifest is not None:
        recipe["contamination_model_manifest"] = str(contamination_model_manifest)
    return recipe


def build_persistent_mps_minus_pair_recipe(
    *,
    contamination_model_manifest: Path | str | None = None,
) -> dict[str, Any]:
    route_id = PERSISTENT_MPS_MINUS_PAIR_ROUTE_ID
    command_template = _vocal_route_command_template(route_id)
    shared_cache_id = f"{route_id}:vocals_minus_pair"

    def build_candidate(candidate_id: str, repair: Mapping[str, Any] | None = None) -> dict[str, Any]:
        candidate: dict[str, Any] = {
            "candidate_id": candidate_id,
            "profile": PERSISTENT_MPS_MINUS_PAIR_PROFILE,
            "route_family": "persistent_mps_dedicated_minus_pair",
            "minus_source": "minus_lane",
            "minus_rebuilt_from_non_vocal_stems": False,
            "diagnostic_only": True,
            "required_stems": ["vocals", "minus"],
            "contamination_check_mode": "explicit_only",
            "use_default_contamination_model_manifest": False,
            "contamination_checks": [{"tested_stem": "minus", "foreign_source": "vocals"}],
            "cross_stem_forensic_gate": {
                "enabled": True,
                "pairs": [{"tested_stem": "minus", "source_stem": "vocals"}],
            },
            "target_specific_commands": [
                {
                    "role": "vocals",
                    "lane": "vocal_lane",
                    "route_id": route_id,
                    "command_template": command_template,
                    "server_stems": ["vocals", "instrumental"],
                    "shared_separator_cache_id": shared_cache_id,
                },
                {
                    "role": "minus",
                    "lane": "minus_lane",
                    "route_id": route_id,
                    "command_template": command_template,
                    "output_map": {"minus": "instrumental"},
                    "server_stems": ["vocals", "instrumental"],
                    "shared_separator_cache_id": shared_cache_id,
                },
            ],
            "delivery_pair_reconstruction": {
                "max_delivery_pair_residual_ratio": 0.08,
            },
        }
        if isinstance(repair, Mapping):
            candidate.update(dict(repair))
        return candidate

    candidates = [
        build_candidate("c001"),
        build_candidate(
            "c002",
            {
                "vocal_projection_debleed": {
                    "target_stems": ["minus"],
                    "strength": 1.0,
                    "max_projection_coeff": 1.0,
                },
            },
        ),
        build_candidate(
            "c003",
            {
                "post_minus_vocal_debleed": {
                    "target_stems": ["minus"],
                    "strength": 0.85,
                    "low_hz": 700.0,
                    "high_hz": 1200.0,
                    "passes": 1,
                },
            },
        ),
        build_candidate(
            "c004",
            {
                "post_minus_vocal_debleed": {
                    "target_stems": ["minus"],
                    "strength": 1.0,
                    "low_hz": 700.0,
                    "high_hz": 1200.0,
                    "passes": 1,
                },
            },
        ),
    ]
    recipe: dict[str, Any] = {
        "schema": "stem_search_recipe.v1",
        "profile": PERSISTENT_MPS_MINUS_PAIR_PROFILE,
        "recipe_source": "persistent_mps_dedicated_minus_pair",
        "available": True,
        "required_stems": ["vocals", "minus"],
        "target_specific_routes": {
            "vocal_lane": [route_id],
            "minus_lane": [route_id],
        },
        "timeout_sec": 900,
        "candidates": candidates,
    }
    if contamination_model_manifest is not None:
        recipe["contamination_model_manifest"] = str(contamination_model_manifest)
    return recipe


def build_default_from_scratch_recipe(
    *,
    demucs_available: bool | None = None,
    available_vocal_routes: tuple[str, ...] | None = None,
    contamination_model_manifest: Path | str | None = None,
) -> dict[str, Any]:
    if demucs_available is None:
        demucs_available = shutil.which("demucs") is not None
    if not demucs_available:
        return {
            "schema": "stem_search_recipe.v1",
            "recipe_source": "from_scratch_default",
            "available": False,
            "unavailable_reason": "missing_demucs_cli",
            "candidates": [],
        }
    vocal_routes = _available_default_vocal_routes(available_vocal_routes)
    if not vocal_routes:
        return {
            "schema": "stem_search_recipe.v1",
            "recipe_source": "from_scratch_default",
            "available": False,
            "unavailable_reason": "missing_target_specific_vocal_route",
            "candidates": [],
        }
    variant_specs = (
        ("c001", "6", "0.25", "1", "", 0.0),
        ("c002", "6", "0.50", "1", "", 0.0),
        ("c003", "6", "0.25", "2", "", 0.0),
        ("c004", "5", "0.50", "1", "", 0.0),
        ("c005", "6", "0.25", "1", "other", 0.0),
        ("c006", "6", "0.25", "1", "", 0.65),
        ("c007", "6", "0.50", "1", "", 0.8),
        ("c008", "6", "0.25", "2", "", 0.9),
        ("c009", "5", "0.50", "1", "", 1.0),
        ("c010", "6", "0.25", "1", "other", 1.0),
    )

    def demucs_candidate(
        candidate_id: str,
        segment: str,
        overlap: str,
        shifts: str,
        residual_stem: str,
        debleed_strength: float,
    ) -> dict[str, Any]:
        route_index = (int(candidate_id[1:]) - 1) % len(vocal_routes)
        vocal_route_id = vocal_routes[route_index]
        candidate = {
            "candidate_id": candidate_id,
            "route_family": f"target_specific_demucs_6s_{vocal_route_id}",
            "vocal_route_id": vocal_route_id,
            "residual_correction_stem": residual_stem,
            "conservation_repair_stem": "other",
            "max_conservation_repair_ratio": 0.30,
            "required_stems": list(STEMS),
            "target_specific_commands": [
                {
                    "role": "six_stems",
                    "route_id": "demucs_htdemucs_6s",
                    "command_template": _demucs_six_stem_command_template(
                        segment=segment,
                        overlap=overlap,
                        shifts=shifts,
                        residual_stem=residual_stem,
                    ),
                },
                {
                    "role": "four_stem_non_vocal",
                    "route_id": "scnet_xl_ihf_musdb18_non_vocal",
                    "command_template": [
                        "python",
                        "tools/run_registered_non_vocal_candidate.py",
                        "--input-audio",
                        "{input_audio}",
                        "--stems-dir",
                        "{stems_dir}",
                        "--work-dir",
                        "{output_dir}/work",
                        "--manifest",
                        "{output_dir}/registered_non_vocal.json",
                        "--model-id",
                        "scnet_xl_ihf_musdb18",
                        "--device",
                        "cuda",
                        "--timeout-sec",
                        "900",
                    ],
                },
                {
                    "role": "vocals",
                    "route_id": vocal_route_id,
                    "command_template": _vocal_route_command_template(vocal_route_id),
                },
            ],
        }
        if debleed_strength > 0.0:
            candidate["minus_vocal_debleed"] = {
                "strength": debleed_strength,
                "low_hz": 700.0,
                "high_hz": 1200.0,
                "target_stems": list(NON_VOCAL_STEMS),
                "passes": 2 if candidate_id == "c010" else 1,
            }
            candidate["vocal_projection_debleed"] = {
                "strength": debleed_strength,
                "target_stems": list(NON_VOCAL_STEMS),
            }
        return candidate

    def instrumental_demucs_candidate(
        candidate_id: str,
        segment: str,
        overlap: str,
        shifts: str,
        debleed_strength: float,
        vocal_model_filename: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        vocal_model_file_dir: str = "/app/models/audio_separator",
        vocal_model_id: str = "audio_separator_bs_roformer_vocals",
        vocal_model_display_name: str = "Audio Separator BS-RoFormer vocal checkpoint",
        instrumental_model_id: str = "",
    ) -> dict[str, Any]:
        candidate = {
            "candidate_id": candidate_id,
            "route_family": f"vocal_instrumental_demucs_6s_{vocal_model_id}",
            "vocal_route_id": vocal_model_id,
            "conservation_repair_stem": "other",
            "max_conservation_repair_ratio": 0.30,
            "required_stems": list(STEMS),
            "command_template": [
                "python",
                "tools/run_vocal_instrumental_demucs_candidate.py",
                "--input-audio",
                "{input_audio}",
                "--stems-dir",
                "{stems_dir}",
                "--work-dir",
                "{output_dir}/work",
                "--manifest",
                "{output_dir}/vdemucs.json",
                "--vocal-model-filename",
                vocal_model_filename,
                "--vocal-model-file-dir",
                vocal_model_file_dir,
                "--vocal-model-id",
                vocal_model_id,
                "--vocal-model-display-name",
                vocal_model_display_name,
                "--instrumental-model-id",
                instrumental_model_id,
                "--segment-sec",
                segment,
                "--overlap",
                overlap,
                "--shifts",
                shifts,
                "--device",
                "cuda",
                "--timeout-sec",
                "900",
            ],
        }
        if debleed_strength > 0.0:
            candidate["minus_vocal_debleed"] = {
                "strength": debleed_strength,
                "low_hz": 700.0,
                "high_hz": 1200.0,
                "target_stems": list(NON_VOCAL_STEMS),
                "passes": 1,
            }
            candidate["vocal_projection_debleed"] = {
                "strength": debleed_strength,
                "target_stems": list(NON_VOCAL_STEMS),
            }
        return candidate

    def registered_six_stem_candidate(
        candidate_id: str,
        model_id: str,
        debleed_strength: float,
        conservation_repair_stem: str = "other",
    ) -> dict[str, Any]:
        candidate = {
            "candidate_id": candidate_id,
            "route_family": f"registered_six_stem_{model_id}",
            "registered_model_id": model_id,
            "conservation_repair_stem": conservation_repair_stem,
            "max_conservation_repair_ratio": 0.30,
            "required_stems": list(STEMS),
            "command_template": [
                "python",
                "tools/run_registered_six_stem_candidate.py",
                "--input-audio",
                "{input_audio}",
                "--stems-dir",
                "{stems_dir}",
                "--work-dir",
                "{output_dir}/work",
                "--manifest",
                "{output_dir}/registered.json",
                "--model-id",
                model_id,
                "--device",
                "cuda",
                "--timeout-sec",
                "900",
            ],
        }
        if debleed_strength > 0.0:
            candidate["minus_vocal_debleed"] = {
                "strength": debleed_strength,
                "low_hz": 700.0,
                "high_hz": 1200.0,
                "target_stems": list(NON_VOCAL_STEMS),
                "passes": 2 if candidate_id == "c010" else 1,
            }
            candidate["vocal_projection_debleed"] = {
                "strength": debleed_strength,
                "target_stems": list(NON_VOCAL_STEMS),
            }
        return candidate

    recipe = {
        "schema": "stem_search_recipe.v1",
        "recipe_source": "from_scratch_default",
        "available": True,
        "target_specific_routes": {
            "six_stems": "demucs_htdemucs_6s",
            "vocals": list(vocal_routes),
        },
        "timeout_sec": 900,
        "candidates": [
            instrumental_demucs_candidate(
                spec[0],
                spec[1],
                spec[2],
                spec[3],
                spec[5],
                *(
                    (
                        "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
                        "/app/vendor/melband_roformer_viperx_11436",
                        "audio_separator_melband_roformer_viperx_vocals",
                        "Audio Separator MelBand RoFormer ViperX vocal checkpoint",
                        "becruily_instrumental_vocals",
                    )
                    if spec[0] == "c010"
                    else ()
                ),
            )
            if spec[0] == "c009"
            else registered_six_stem_candidate(
                spec[0],
                "hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first",
                spec[5],
                "guitar",
            )
            if spec[0] == "c010"
            else demucs_candidate(*spec)
            for spec in variant_specs
        ],
    }
    for candidate in recipe["candidates"]:
        if candidate["candidate_id"] == "c010":
            candidate["stem_merges"] = [
                {
                    "source_stem": "other",
                    "target_stem": "guitar",
                    "mode": "move",
                    "reason": "aggressive_melody_first_harmonic_bus",
                }
            ]
            candidate["post_minus_vocal_debleed"] = {
                "strength": 1.0,
                "low_hz": 700.0,
                "high_hz": 1200.0,
                "target_stems": list(NON_VOCAL_STEMS),
                "passes": 1,
            }
    if contamination_model_manifest is not None:
        recipe["contamination_model_manifest"] = str(contamination_model_manifest)
    return recipe


def split_audio_windows(mix_path: Path, output_dir: Path, window_sec: float) -> list[dict[str, Any]]:
    audio, sample_rate = _read_audio(mix_path)
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")
    samples_per_window = max(1, int(round(window_sec * sample_rate)))
    min_trailing_samples = max(1, int(round(MIN_TRAILING_WINDOW_SEC * sample_rate)))
    segments: list[tuple[int, int, dict[str, Any]]] = []
    for start in range(0, len(audio), samples_per_window):
        end = min(len(audio), start + samples_per_window)
        if end <= start:
            continue
        segments.append((start, end, {}))
    if len(segments) >= 2:
        tail_start, tail_end, _tail_meta = segments[-1]
        tail_samples = tail_end - tail_start
        if tail_samples < min_trailing_samples:
            prev_start, _prev_end, prev_meta = segments[-2]
            merged_meta = dict(prev_meta)
            merged_meta.update(
                {
                    "short_tail_merged": True,
                    "merged_tail_sec": round(tail_samples / sample_rate, 6),
                    "merged_tail_sample_count": int(tail_samples),
                    "nominal_window_sec": float(window_sec),
                }
            )
            segments[-2] = (prev_start, tail_end, merged_meta)
            segments.pop()
    windows: list[dict[str, Any]] = []
    for start, end, segment_meta in segments:
        window_id = f"w{len(windows):03d}"
        window_path = _write_audio(output_dir / window_id / "mix.flac", audio[start:end], sample_rate)
        window = {
            "window_id": window_id,
            "mix_path": str(window_path),
            "start_sec": round(start / sample_rate, 6),
            "end_sec": round(end / sample_rate, 6),
            "sample_rate": sample_rate,
            "sample_count": int(end - start),
            "channel_count": int(audio.shape[1]),
            "short_tail_merged": bool(segment_meta.get("short_tail_merged", False)),
        }
        if segment_meta.get("short_tail_merged"):
            window.update(segment_meta)
        windows.append(window)
    return windows


def build_candidate_manifest(candidate_id: str, output_dir: Path) -> dict[str, Any]:
    return {
        "schema": "fully_automatic_stem_candidate.v1",
        "candidate_id": candidate_id,
        "candidate_untrusted": True,
        "verdict": "blocked_unknown",
        "output_dir": str(output_dir),
        "required_stems": list(STEMS),
        "minus_rebuilt_from_non_vocal_stems": False,
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
    }


def _candidate_items(recipe: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items = recipe.get("candidates")
    if not isinstance(items, list):
        items = recipe.get("commands")
    return [item for item in (items or []) if isinstance(item, dict)]


def _target_specific_route_signature(candidate: Mapping[str, Any]) -> str:
    commands = candidate.get("target_specific_commands")
    if not isinstance(commands, list):
        return str(candidate.get("route_signature") or "").strip()
    parts: list[str] = []
    for command in commands:
        if not isinstance(command, Mapping):
            continue
        lane = str(command.get("lane") or command.get("role") or "").strip()
        route_id = str(command.get("route_id") or command.get("route_alias") or "").strip()
        if lane or route_id:
            parts.append(f"{lane}:{route_id}")
    return "|".join(parts)


def _signature_matches(candidate_signature: str, policy_signature: str) -> bool:
    candidate_parts = {part for part in str(candidate_signature or "").split("|") if part}
    policy_parts = {part for part in str(policy_signature or "").split("|") if part}
    if not candidate_parts or not policy_parts:
        return False

    def _part_matches(candidate_part: str, policy_part: str) -> bool:
        return candidate_part == policy_part or candidate_part.startswith(f"{policy_part}:")

    return all(any(_part_matches(candidate_part, policy_part) for candidate_part in candidate_parts) for policy_part in policy_parts)


def _load_root_cause_route_policy(root_cause_queue_jsonl: Path | None) -> dict[str, Any]:
    if root_cause_queue_jsonl is None:
        return {
            "schema": "model_split_root_cause_route_policy.v1",
            "source_queue_jsonl": "",
            "quarantined_route_signatures": [],
            "repairable_route_signatures": [],
            "blocked_repair_ids": [],
            "blocked_repair_checkpoints": [],
        }
    rows = [
        json.loads(line)
        for line in Path(root_cause_queue_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    quarantined: set[str] = set()
    repairable: set[str] = set()
    blocked_repair_ids: set[str] = set()
    blocked_repair_checkpoints: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        failed_repair_id = str(row.get("failed_repair_id") or "").strip()
        if failed_repair_id:
            blocked_repair_ids.add(failed_repair_id)
        failed_repair_checkpoint = str(row.get("failed_repair_checkpoint") or "").strip()
        if failed_repair_checkpoint:
            blocked_repair_checkpoints.add(failed_repair_checkpoint)
        signature = str(row.get("route_signature") or row.get("route_family") or "").strip()
        if not signature:
            continue
        if bool(row.get("quarantined_for_next_run")):
            quarantined.add(signature)
        if str(row.get("issue_type") or "") == ISSUE_BAD_RECONSTRUCTION and str(row.get("affected_stem") or ""):
            repairable.add(signature)
    return {
        "schema": "model_split_root_cause_route_policy.v1",
        "source_queue_jsonl": str(root_cause_queue_jsonl),
        "quarantined_route_signatures": sorted(quarantined),
        "repairable_route_signatures": sorted(repairable),
        "blocked_repair_ids": sorted(blocked_repair_ids),
        "blocked_repair_checkpoints": sorted(blocked_repair_checkpoints),
    }


def _repair_option_checkpoints(option: Mapping[str, Any]) -> set[str]:
    checkpoints: set[str] = set()

    def add_checkpoint(value: Any) -> None:
        checkpoint = str(value or "").strip()
        if checkpoint:
            checkpoints.add(checkpoint)

    add_checkpoint(option.get("checkpoint"))
    config = option.get("config")
    if isinstance(config, Mapping):
        add_checkpoint(config.get("checkpoint") or config.get("checkpoint_path"))
    stem_region_repair = option.get("stem_region_repair")
    if isinstance(stem_region_repair, Mapping):
        add_checkpoint(stem_region_repair.get("checkpoint") or stem_region_repair.get("checkpoint_path"))
    sequence = option.get("stem_region_repairs")
    if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes)):
        for item in sequence:
            if isinstance(item, Mapping):
                add_checkpoint(item.get("checkpoint") or item.get("checkpoint_path"))
    return checkpoints


def _filter_blocked_repair_options(
    item: Mapping[str, Any],
    *,
    blocked_repair_ids: set[str],
    blocked_repair_checkpoints: set[str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    updated = dict(item)
    repair_options = item.get("model_split_composite_repair_options")
    if not isinstance(repair_options, list) or not repair_options:
        return updated, []
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for option in repair_options:
        if not isinstance(option, Mapping):
            continue
        repair_id = str(option.get("repair_id") or "").strip()
        checkpoints = _repair_option_checkpoints(option)
        matched_checkpoint = next((checkpoint for checkpoint in sorted(checkpoints) if checkpoint in blocked_repair_checkpoints), "")
        if (repair_id and repair_id in blocked_repair_ids) or matched_checkpoint:
            skipped.append(
                {
                    "candidate_id": str(item.get("candidate_id") or ""),
                    "repair_id": repair_id,
                    "matched_checkpoint": matched_checkpoint,
                }
            )
            continue
        kept.append(dict(option))
    updated["model_split_composite_repair_options"] = kept
    return updated, skipped


def _apply_root_cause_route_policy(
    recipe: Mapping[str, Any],
    *,
    root_cause_queue_jsonl: Path | None,
) -> dict[str, Any]:
    policy = _load_root_cause_route_policy(root_cause_queue_jsonl)
    items = _candidate_items(recipe)
    quarantined = list(policy.get("quarantined_route_signatures") or [])
    repairable = list(policy.get("repairable_route_signatures") or [])
    blocked_repair_ids = {str(value) for value in policy.get("blocked_repair_ids") or []}
    blocked_repair_checkpoints = {str(value) for value in policy.get("blocked_repair_checkpoints") or []}

    kept: list[Mapping[str, Any]] = []
    skipped: list[dict[str, str]] = []
    skipped_repair_options: list[dict[str, str]] = []
    for item in items:
        signature = _target_specific_route_signature(item)
        matched_quarantine = next((blocked for blocked in quarantined if _signature_matches(signature, blocked)), "")
        if matched_quarantine:
            skipped.append(
                {
                    "candidate_id": str(item.get("candidate_id") or ""),
                    "route_signature": signature,
                    "matched_quarantine": matched_quarantine,
                }
            )
            continue
        filtered_item, repair_skips = _filter_blocked_repair_options(
            item,
            blocked_repair_ids=blocked_repair_ids,
            blocked_repair_checkpoints=blocked_repair_checkpoints,
        )
        skipped_repair_options.extend(repair_skips)
        kept.append(filtered_item)

    def _priority(item: Mapping[str, Any]) -> tuple[int, str]:
        signature = _target_specific_route_signature(item)
        is_repairable = any(_signature_matches(signature, value) for value in repairable)
        return (0 if is_repairable else 1, str(item.get("candidate_id") or ""))

    ordered = sorted(kept, key=_priority)
    updated = dict(recipe)
    updated["candidates"] = [dict(item) for item in ordered]
    policy = dict(policy)
    policy["skipped_candidate_count"] = len(skipped)
    policy["skipped_candidates"] = skipped
    policy["skipped_repair_option_count"] = len(skipped_repair_options)
    policy["skipped_repair_options"] = skipped_repair_options
    policy["prioritized_candidate_count"] = sum(1 for item in ordered if _priority(item)[0] == 0)
    updated["root_cause_route_policy"] = policy
    return updated


def _load_learned_route_policy_checkpoint(route_policy_checkpoint: Path | None) -> dict[str, Any] | None:
    if route_policy_checkpoint is None:
        return None
    return load_stem_route_policy_checkpoint(Path(route_policy_checkpoint))


def _route_policy_row_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    verdict = str(payload.get("verdict") or "")
    issue_type = verdict[8:] if verdict.startswith("blocked_") else verdict
    return {
        "schema": "stem_route_policy_row.v1",
        "row_id": str(payload.get("row_id") or payload.get("candidate_id") or "candidate"),
        "source_run": str(payload.get("source_run") or payload.get("run_json") or ""),
        "window_id": str(payload.get("window_id") or ""),
        "candidate_id": str(payload.get("candidate_id") or ""),
        "route_signature": extract_route_signature(payload),
        "lane_routes": extract_lane_routes(payload),
        "issue_type": str(payload.get("issue_type") or issue_type or "unknown"),
        "root_cause": str(payload.get("root_cause") or ""),
        "affected_stem": str(payload.get("affected_stem") or ""),
        "gate_verdicts": {
            "candidate_verdict": verdict,
            "candidate_delivery_allowed": bool(payload.get("delivery_allowed", False)),
            "candidate_presentation_status": str(payload.get("presentation_status") or ""),
        },
        "forensic_features": payload.get("cross_stem_forensic_score") if isinstance(payload.get("cross_stem_forensic_score"), Mapping) else {},
        "damage_features": payload.get("damage_score") if isinstance(payload.get("damage_score"), Mapping) else {},
        "reconstruction_features": payload.get("reconstruction") if isinstance(payload.get("reconstruction"), Mapping) else {},
        "current_action": ACTION_BLOCK_UNKNOWN,
        "oracle_action": ACTION_BLOCK_UNKNOWN,
        "oracle_label_source": "runtime_scoring_only",
        "usable_for_training": False,
        "usable_for_promotion": False,
        "is_ground_truth": False,
        "label_type": "runtime_candidate",
    }


def _score_learned_route_policy_payload(
    payload: Mapping[str, Any],
    *,
    route_policy_model: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if route_policy_model is None:
        return {
            "schema": "stem_route_policy_score.v1",
            "action": ACTION_BLOCK_UNKNOWN,
            "action_probs": {action: 1.0 if action == ACTION_BLOCK_UNKNOWN else 0.0 for action in ROUTE_POLICY_ACTIONS},
            "score": 0.0,
            "match_level": "disabled",
            "shadow_only": True,
            "delivery_allowed": False,
            "blockers": ["route_policy_disabled"],
        }
    score = dict(score_policy_row(route_policy_model, _route_policy_row_from_payload(payload)))
    action = normalize_policy_action(str(score.get("action") or ""))
    score["action"] = action
    score["delivery_allowed"] = False
    score["shadow_only"] = True
    return score


def _apply_learned_route_policy_to_recipe(
    recipe: Mapping[str, Any],
    *,
    route_policy_model: Mapping[str, Any] | None,
    route_policy_mode: str,
) -> dict[str, Any]:
    mode = str(route_policy_mode or "shadow").strip().lower()
    if mode not in {"shadow", "rank"}:
        mode = "shadow"
    scored_items: list[dict[str, Any]] = []
    for item in _candidate_items(recipe):
        candidate_item = dict(item)
        score = _score_learned_route_policy_payload(candidate_item, route_policy_model=route_policy_model)
        candidate_item["route_policy_preview"] = {
            "action": score["action"],
            "score": score["score"],
            "match_level": score["match_level"],
            "shadow_only": True,
            "blockers": score.get("blockers") or [],
        }
        scored_items.append(candidate_item)

    if mode == "rank":
        def _rank_key(item: Mapping[str, Any]) -> tuple[int, float, str]:
            preview = item.get("route_policy_preview") if isinstance(item.get("route_policy_preview"), Mapping) else {}
            action = normalize_policy_action(str(preview.get("action") or ""))
            score = float(preview.get("score") or 0.0)
            if action == ACTION_PRIORITIZE_ROUTE:
                priority = 0
            elif action == ACTION_QUARANTINE_ROUTE:
                priority = 8
            elif action == ACTION_BLOCK_UNKNOWN:
                priority = 9
            else:
                priority = 1
            return (priority, -score, str(item.get("candidate_id") or ""))

        scored_items = sorted(scored_items, key=_rank_key)

    updated = dict(recipe)
    updated["candidates"] = scored_items
    updated["learned_route_policy"] = {
        "schema": "stem_route_policy_runtime.v1",
        "mode": mode,
        "shadow_only": mode == "shadow",
        "can_accept_delivery": False,
        "candidate_count": len(scored_items),
    }
    return updated


def _annotate_candidate_with_learned_route_policy(
    candidate_manifest: Mapping[str, Any],
    *,
    route_policy_model: Mapping[str, Any] | None,
    route_policy_mode: str,
) -> dict[str, Any]:
    annotated = dict(candidate_manifest)
    score = _score_learned_route_policy_payload(annotated, route_policy_model=route_policy_model)
    action = normalize_policy_action(str(score.get("action") or ""))
    blockers = list(score.get("blockers") or [])
    annotated.update(
        {
            "route_policy_score": float(score.get("score") or 0.0),
            "route_policy_action": action,
            "route_policy_action_probs": score.get("action_probs") or {},
            "route_policy_match_level": score.get("match_level") or "",
            "route_policy_shadow_only": True,
            "route_policy_mode": str(route_policy_mode or "shadow"),
            "route_policy_blockers": blockers,
        }
    )
    if action not in ROUTE_POLICY_ACTIONS:
        annotated["verdict"] = "blocked_unknown"
        annotated["delivery_allowed"] = False
        annotated["presentation_status"] = "blocked_from_listen_pack"
        annotated["route_policy_blockers"] = [*blockers, "invalid_policy_action"]
    if annotated.get("verdict") != "auto_accept":
        annotated["delivery_allowed"] = False
        annotated["presentation_status"] = "blocked_from_listen_pack"
    return annotated


def _persist_candidate_manifest(candidate_manifest: Mapping[str, Any]) -> None:
    output_dir = str(candidate_manifest.get("output_dir") or "").strip()
    if output_dir:
        _write_json(Path(output_dir) / "manifest.json", dict(candidate_manifest))


def _target_specific_items(candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items = candidate.get("target_specific_commands")
    return [item for item in (items or []) if isinstance(item, dict)] if isinstance(items, list) else []


def _split_model_split_v2_delivery_pair_items(
    candidate: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    delivery_items: list[Mapping[str, Any]] = []
    remaining_items: list[Mapping[str, Any]] = []
    for item in _target_specific_items(candidate):
        role = str(item.get("role") or "").strip()
        if role in MODEL_SPLIT_V2_DELIVERY_PAIR_ROLES:
            delivery_items.append(item)
        else:
            remaining_items.append(item)
    return delivery_items, remaining_items


def _can_model_split_v2_delivery_pair_fail_fast(candidate: Mapping[str, Any]) -> bool:
    if not _is_model_split_v2(candidate):
        return False
    if candidate.get("delivery_pair_fail_fast") is not True:
        return False
    delivery_items, remaining_items = _split_model_split_v2_delivery_pair_items(candidate)
    if not delivery_items or not remaining_items:
        return False
    roles = {str(item.get("role") or "").strip() for item in delivery_items}
    if not MODEL_SPLIT_V2_DELIVERY_PAIR_ROLES.issubset(roles):
        return False
    return not any(candidate.get(key) for key in MODEL_SPLIT_V2_FAIL_FAST_MUTATION_KEYS)


def _with_target_specific_items(candidate: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    patched = dict(candidate)
    patched["target_specific_commands"] = [dict(item) for item in items]
    return patched


def _merge_target_specific_compositions(*compositions: Mapping[str, Any]) -> dict[str, Any]:
    routes: list[dict[str, Any]] = []
    blocker_reasons: list[str] = []
    for composition in compositions:
        routes.extend(dict(route) for route in (composition.get("routes") or []) if isinstance(route, Mapping))
        blocker_reasons.extend(str(reason) for reason in (composition.get("blocker_reasons") or []))
    return {
        "applied": any(bool(composition.get("applied")) for composition in compositions),
        "routes": routes,
        "blocker_reasons": blocker_reasons,
    }


def _route_timeout_sec(
    item: Mapping[str, Any],
    route_id: str,
    default_timeout_sec: int,
    *,
    route_budget_profile: str = ROUTE_BUDGET_PROFILE_FULL,
) -> int:
    raw = item.get("route_timeout_sec", item.get("timeout_sec"))
    if raw is not None:
        try:
            return max(1, int(float(raw)))
        except (TypeError, ValueError):
            return max(1, int(default_timeout_sec))
    normalized_profile = str(route_budget_profile or ROUTE_BUDGET_PROFILE_FULL).strip().lower()
    if normalized_profile == ROUTE_BUDGET_PROFILE_FAST:
        timeout_overrides = {**ROUTE_TIMEOUT_OVERRIDES_SEC, **FAST_ROUTE_TIMEOUT_OVERRIDES_SEC}
    else:
        timeout_overrides = ROUTE_TIMEOUT_OVERRIDES_SEC
    return max(1, int(timeout_overrides.get(str(route_id or ""), int(default_timeout_sec))))


def _collect_stem_paths(stems_dir: Path) -> dict[str, str]:
    return {stem: str(stems_dir / f"{stem}.flac") for stem in [*STEMS, "minus"] if (stems_dir / f"{stem}.flac").is_file()}


def _candidate_required_stems(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    raw = candidate.get("required_stems")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return tuple(STEMS)
    stems: list[str] = []
    allowed = {*STEMS, "minus"}
    for value in raw:
        stem = str(value or "").strip()
        if stem not in allowed:
            continue
        if stem not in stems:
            stems.append(stem)
    return tuple(stems or STEMS)


def _missing_candidate_required_stems(candidate: Mapping[str, Any], stem_paths: Mapping[str, Any]) -> list[str]:
    present = set(stem_paths)
    return [stem for stem in _candidate_required_stems(candidate) if stem not in present]


def _safe_artifact_segment(value: str, *, fallback: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._-")
    if not segment:
        segment = fallback
    return segment[:64]


def _copy_target_specific_outputs(
    *,
    role: str,
    role_stems_dir: Path,
    final_stems_dir: Path,
    target_sample_rate: int | None = None,
    target_sample_count: int | None = None,
    target_channel_count: int | None = None,
    output_map: Mapping[str, str] | None = None,
    copied_outputs: list[dict[str, str]] | None = None,
) -> list[str]:
    required_outputs = TARGET_SPECIFIC_ROLE_OUTPUTS.get(role)
    if required_outputs is None:
        return [f"unsupported_target_specific_role:{role}"]
    mapped_outputs = {str(target): str(source) for target, source in (output_map or {}).items()}
    blockers: list[str] = []
    for stem in required_outputs:
        source_stem = mapped_outputs.get(stem, stem)
        source_path = role_stems_dir / f"{source_stem}.flac"
        if not source_path.is_file():
            blockers.append(f"missing_target_specific_output:{role}:{stem}")
            continue
        final_stems_dir.mkdir(parents=True, exist_ok=True)
        output_path = final_stems_dir / f"{stem}.flac"
        _copy_audio_to_match_shape(
            source_path,
            output_path,
            target_sample_rate=target_sample_rate,
            target_sample_count=target_sample_count,
            target_channel_count=target_channel_count,
        )
        if copied_outputs is not None:
            copied_outputs.append(
                {
                    "stem": stem,
                    "source_stem": source_stem,
                    "source_path": str(source_path),
                    "output_path": str(output_path),
                }
            )
    return blockers


def _run_target_specific_commands(
    *,
    candidate: Mapping[str, Any],
    window: Mapping[str, Any],
    candidate_dir: Path,
    stems_dir: Path,
    command_runner: CommandRunner,
    timeout_sec: int,
    route_cache: dict[str, dict[str, Any]] | None = None,
    separator_backend_config: SeparatorBackendConfig | None = None,
    separator_client: PersistentSeparatorClient | None = None,
    route_budget_profile: str = ROUTE_BUDGET_PROFILE_FULL,
) -> dict[str, Any]:
    backend_config = (separator_backend_config or SeparatorBackendConfig()).validated()
    persistent_client = separator_client
    routes: list[dict[str, Any]] = []
    blocker_reasons: list[str] = []
    for idx, item in enumerate(_target_specific_items(candidate), start=1):
        role = str(item.get("role") or "").strip()
        command_template = item.get("command_template")
        route_id = str(item.get("route_id") or role or f"r{idx:03d}")
        route_alias = _safe_artifact_segment(route_id, fallback=f"r{idx:03d}")
        lane = str(item.get("lane") or role or route_id).strip()
        prefer_resident = bool(item.get("prefer_resident")) or _prefers_resident_separator(route_id)
        route_timeout_sec = _route_timeout_sec(item, route_id, timeout_sec, route_budget_profile=route_budget_profile)
        output_map = item.get("output_map") if isinstance(item.get("output_map"), dict) else {}
        route_dir = candidate_dir / "routes" / route_alias
        route_stems_dir = route_dir / "stems"
        route_manifest = {
            "role": role,
            "lane": lane,
            "route_id": route_id,
            "route_alias": route_alias,
            "source_candidate_id": str(candidate.get("candidate_id") or ""),
            "output_dir": str(route_dir),
            "stems_dir": str(route_stems_dir),
            "required_outputs": list(TARGET_SPECIFIC_ROLE_OUTPUTS.get(role, ())),
            "output_map": dict(output_map),
            "copied_outputs": [],
            "missing_outputs": [],
            "blocker_reasons": [],
            "timeout_sec": route_timeout_sec,
            "separator_backend": {
                **backend_config.manifest(),
                "cache_hit": False,
                "route_cache_key": "",
                "prefer_resident": prefer_resident,
                "route_budget_profile": str(route_budget_profile or ROUTE_BUDGET_PROFILE_FULL),
                "route_timeout_sec": route_timeout_sec,
            },
        }
        if not isinstance(command_template, list) or not command_template:
            route_manifest["blocker_reasons"] = [f"missing_target_specific_command:{role or route_id}"]
            blocker_reasons.extend(route_manifest["blocker_reasons"])
            routes.append(route_manifest)
            continue
        replacements = {
            "window_id": str(window["window_id"]),
            "input_audio": str(window["mix_path"]),
            "window_mix": str(window["mix_path"]),
            "output_dir": str(route_dir),
            "stems_dir": str(route_stems_dir),
            "candidate_id": str(candidate.get("candidate_id") or "c001"),
            "route_role": role,
            "route_id": route_id,
        }
        command = render_recipe_command(command_template, replacements)
        route_manifest["command"] = command
        required_outputs = list(TARGET_SPECIFIC_ROLE_OUTPUTS.get(role, ()))
        server_stems = item.get("server_stems") if isinstance(item.get("server_stems"), list) else []
        server_required_outputs = (
            list(dict.fromkeys(str(stem) for stem in server_stems if str(stem).strip()))
            if server_stems
            else list(dict.fromkeys(str(output_map.get(stem, stem)) for stem in required_outputs))
        )
        route_manifest["server_required_outputs"] = list(server_required_outputs)
        shared_cache_id = str(item.get("shared_separator_cache_id") or "").strip()
        cache_params: dict[str, Any] = {
            "backend": backend_config.mode,
            "prefer_resident": prefer_resident,
            "route_budget_profile": str(route_budget_profile or ROUTE_BUDGET_PROFILE_FULL),
            "route_timeout_sec": route_timeout_sec,
        }
        if shared_cache_id:
            cache_params["shared_separator_cache_id"] = shared_cache_id
        else:
            cache_params.update(
                {
                    "role": role,
                    "lane": lane,
                    "output_map": dict(output_map),
                }
            )
        cache_key = separator_cache_key(
            route_id=route_id,
            source_audio_path=str(window["mix_path"]),
            command=[str(part) for part in command_template],
            stems=server_required_outputs,
            params=cache_params,
        )
        route_manifest["route_cache_key"] = cache_key
        route_manifest["separator_backend"]["route_cache_key"] = cache_key
        cached_route = (route_cache or {}).get(cache_key)
        if cached_route:
            cached_stems_dir = Path(str(cached_route.get("stems_dir") or ""))
            copied_outputs: list[dict[str, str]] = []
            route_blockers = _copy_target_specific_outputs(
                role=role,
                role_stems_dir=cached_stems_dir,
                final_stems_dir=stems_dir,
                target_sample_rate=int(window["sample_rate"]) if window.get("sample_rate") is not None else None,
                target_sample_count=int(window["sample_count"]) if window.get("sample_count") is not None else None,
                target_channel_count=int(window["channel_count"]) if window.get("channel_count") is not None else None,
                output_map=output_map,
                copied_outputs=copied_outputs,
            )
            route_manifest.update(
                {
                    "reused_route": True,
                    "reused_from_candidate_id": str(cached_route.get("source_candidate_id") or ""),
                    "reused_from_output_dir": str(cached_route.get("output_dir") or ""),
                    "reused_from_stems_dir": str(cached_stems_dir),
                    "copied_outputs": copied_outputs,
                    "missing_outputs": [
                        reason.split(":")[-1] for reason in route_blockers if reason.startswith("missing_target_specific_output:")
                    ],
                    "blocker_reasons": route_blockers,
                    "separator_backend": {
                        **route_manifest["separator_backend"],
                        "cache_hit": True,
                        "reused_route": True,
                        "resident_backend_used": bool(cached_route.get("resident_backend_used", False)),
                        "resident_executor": cached_route.get("resident_executor"),
                        "resident_benchmark": cached_route.get("resident_benchmark"),
                    },
                }
            )
            blocker_reasons.extend(route_blockers)
            routes.append(route_manifest)
            continue
        route_stems_dir.mkdir(parents=True, exist_ok=True)
        produced_stems_dir = route_stems_dir
        if backend_config.enabled:
            if persistent_client is None:
                request_timeout_sec = (
                    backend_config.request_timeout_sec
                    if backend_config.request_timeout_sec > 0
                    else float(timeout_sec) + 30.0
                )
                persistent_client = PersistentSeparatorClient(
                    backend_config.server_url,
                    request_timeout_sec=request_timeout_sec,
                )
            response = persistent_client.separate(
                route_id=route_id,
                source_audio_path=str(window["mix_path"]),
                output_dir=str(route_dir),
                device=backend_config.device,
                stems=server_required_outputs,
                batch_id=f"{window['window_id']}:{candidate.get('candidate_id') or 'c001'}:{route_alias}",
                command=command,
                cache_key=cache_key,
                timeout_sec=route_timeout_sec,
                require_gpu=backend_config.require_gpu,
                prefer_resident=prefer_resident,
            )
            produced_stems_dir = Path(str(response.get("stems_dir") or route_stems_dir))
            route_manifest["separator_backend"].update(
                {
                    "cache_hit": bool(response.get("cache_hit")),
                    "server_status": str(response.get("status") or "ok"),
                    "response_manifest_path": str(response.get("manifest_path") or ""),
                    "produced_stems_dir": str(produced_stems_dir),
                    "resident_backend_used": bool(response.get("resident_backend_used", False)),
                    "resident_executor": response.get("resident_executor"),
                    "resident_benchmark": response.get("resident_benchmark"),
                    "gpu_inference_skipped": response.get("gpu_inference_skipped"),
                    "accelerator_mode": response.get("accelerator_mode"),
                }
            )
        else:
            setattr(command_runner, "stems_dir", route_stems_dir)
            command_runner(command, route_timeout_sec)
        copied_outputs: list[dict[str, str]] = []
        route_blockers = _copy_target_specific_outputs(
            role=role,
            role_stems_dir=produced_stems_dir,
            final_stems_dir=stems_dir,
            target_sample_rate=int(window["sample_rate"]) if window.get("sample_rate") is not None else None,
            target_sample_count=int(window["sample_count"]) if window.get("sample_count") is not None else None,
            target_channel_count=int(window["channel_count"]) if window.get("channel_count") is not None else None,
            output_map=output_map,
            copied_outputs=copied_outputs,
        )
        route_manifest["copied_outputs"] = copied_outputs
        route_manifest["missing_outputs"] = [
            reason.split(":")[-1] for reason in route_blockers if reason.startswith("missing_target_specific_output:")
        ]
        route_manifest["blocker_reasons"] = route_blockers
        blocker_reasons.extend(route_blockers)
        routes.append(route_manifest)
        if route_cache is not None and not route_blockers:
            route_cache[cache_key] = {
                "source_candidate_id": str(candidate.get("candidate_id") or ""),
                "output_dir": str(route_dir),
                "stems_dir": str(produced_stems_dir),
                "route_id": route_id,
                "role": role,
                "lane": lane,
                "resident_backend_used": bool(route_manifest["separator_backend"].get("resident_backend_used", False)),
                "resident_executor": route_manifest["separator_backend"].get("resident_executor"),
                "resident_benchmark": route_manifest["separator_backend"].get("resident_benchmark"),
            }
    return {
        "applied": True,
        "routes": routes,
        "blocker_reasons": blocker_reasons,
    }


def _full_forbidden_contamination_checks() -> list[dict[str, str]]:
    return [
        {"tested_stem": tested_stem, "foreign_source": foreign_source}
        for tested_stem, forbidden_sources in FORBIDDEN_SOURCES.items()
        for foreign_source in forbidden_sources
    ]


def _candidate_contamination_checks(candidate: Mapping[str, Any]) -> tuple[list[dict[str, str]], bool]:
    required_checks = _full_forbidden_contamination_checks()
    use_default_model_manifest = candidate.get("use_default_contamination_model_manifest", True) is not False
    default_model_manifest = (
        str(candidate.get("contamination_model_manifest") or "").strip() if use_default_model_manifest else ""
    )
    checks = candidate.get("contamination_checks")
    if not isinstance(checks, list):
        if not default_model_manifest:
            return required_checks, False
        return [{**check, "model_manifest": default_model_manifest} for check in required_checks], True

    explicit_only = str(candidate.get("contamination_check_mode") or "").strip() == "explicit_only"
    hints: dict[tuple[str, str], dict[str, str]] = {}
    explicit_candidates: list[dict[str, str]] = []
    for item in checks:
        if not isinstance(item, dict):
            continue
        tested_stem = str(item.get("tested_stem") or "").strip()
        foreign_source = str(item.get("foreign_source") or "").strip()
        if not tested_stem or not foreign_source:
            continue
        hint: dict[str, str] = {}
        model_manifest = str(item.get("model_manifest") or "").strip()
        if model_manifest:
            hint["model_manifest"] = model_manifest
        hints[(tested_stem, foreign_source)] = hint
        explicit_check = {"tested_stem": tested_stem, "foreign_source": foreign_source}
        if default_model_manifest:
            explicit_check["model_manifest"] = default_model_manifest
        explicit_check.update(hint)
        explicit_candidates.append(explicit_check)

    if explicit_only:
        return explicit_candidates, True

    merged: list[dict[str, str]] = []
    for check in required_checks:
        pair = (check["tested_stem"], check["foreign_source"])
        merged_check = dict(check)
        if default_model_manifest:
            merged_check["model_manifest"] = default_model_manifest
        merged_check.update(hints.get(pair, {}))
        merged.append(merged_check)
    return merged, True


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def validate_mix_conservation(
    *,
    mix_path: Path,
    stem_paths: Mapping[str, str],
    required_stems: Sequence[str] | None = None,
    relative_tolerance: float = 0.01,
) -> dict[str, Any]:
    mix_audio, mix_rate = _read_audio(mix_path)
    summed = np.zeros_like(mix_audio, dtype=np.float32)
    blockers: list[str] = []
    stems = tuple(required_stems or STEMS)
    for stem in stems:
        stem_path = stem_paths.get(stem)
        if not stem_path:
            blockers.append(f"missing_reconstruction_stem:{stem}")
            continue
        stem_audio, stem_rate = _read_audio(Path(stem_path))
        if stem_rate != mix_rate:
            blockers.append(f"sample_rate_mismatch:{stem}:{stem_rate}!={mix_rate}")
            continue
        if stem_audio.shape != mix_audio.shape:
            blockers.append(f"shape_mismatch:{stem}:{stem_audio.shape}!={mix_audio.shape}")
            continue
        summed = summed + stem_audio

    residual = mix_audio - summed
    mix_rms = _rms(mix_audio)
    residual_rms = _rms(residual)
    tolerance = max(1e-5, mix_rms * float(relative_tolerance))
    if residual_rms > tolerance:
        blockers.append("mix_conservation_residual_too_high")
    return {
        "ok": not blockers,
        "blockers": blockers,
        "required_stems": list(stems),
        "relative_tolerance": round(float(relative_tolerance), 8),
        "residual_ratio": round(float(residual_rms / max(mix_rms, 1e-12)), 8),
        "mix_rms": round(mix_rms, 8),
        "residual_rms": round(residual_rms, 8),
        "tolerance": round(tolerance, 8),
    }


def _first_route_stems_dir(target_specific_composition: Mapping[str, Any], role: str) -> Path | None:
    for route in target_specific_composition.get("routes", []) or []:
        if str(route.get("role") or "") != role:
            continue
        raw_stems_dir = route.get("stems_dir")
        if raw_stems_dir:
            return Path(str(raw_stems_dir))
    return None


def _six_stem_lane_paths(target_specific_composition: Mapping[str, Any]) -> dict[str, str]:
    stems_dir = _first_route_stems_dir(target_specific_composition, "six_stems")
    if stems_dir is None:
        return {}
    return {
        stem: str(stems_dir / f"{stem}.flac")
        for stem in STEMS
        if (stems_dir / f"{stem}.flac").exists()
    }


def validate_model_split_v2_conservation(
    *,
    mix_path: Path,
    stem_paths: Mapping[str, str],
    target_specific_composition: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    max_residual_ratio = float(config.get("max_delivery_pair_residual_ratio", 0.06))
    delivery_pair = validate_mix_conservation(
        mix_path=mix_path,
        stem_paths=stem_paths,
        required_stems=("vocals", "minus"),
        relative_tolerance=max_residual_ratio,
    )
    six_stem_paths = _six_stem_lane_paths(target_specific_composition)
    six_stem_diagnostic = (
        validate_mix_conservation(
            mix_path=mix_path,
            stem_paths=six_stem_paths,
            required_stems=STEMS,
            relative_tolerance=float(config.get("six_stem_diagnostic_residual_ratio", 0.01)),
        )
        if six_stem_paths
        else {
            "ok": False,
            "blockers": ["missing_six_stem_lane_diagnostic"],
            "required_stems": list(STEMS),
        }
    )
    return {
        "ok": bool(delivery_pair.get("ok")),
        "mode": "model_split_v2_delivery_pair",
        "gated_surface": "vocals_plus_minus",
        "delivery_pair": delivery_pair,
        "six_stem_diagnostic": six_stem_diagnostic,
    }


def apply_mix_conservation_repair(
    *,
    mix_path: Path,
    stem_paths: Mapping[str, str],
    target_stem: str,
    max_residual_ratio: float | None = None,
) -> dict[str, Any]:
    if target_stem not in NON_VOCAL_STEMS:
        return {
            "applied": False,
            "repair_owner": "runner_conservation_repair_v1",
            "reason": "unsupported_conservation_repair_stem",
            "target_stem": target_stem,
        }
    target_path = stem_paths.get(target_stem)
    if not target_path:
        return {
            "applied": False,
            "repair_owner": "runner_conservation_repair_v1",
            "reason": "missing_conservation_repair_stem",
            "target_stem": target_stem,
        }

    mix_audio, mix_rate = _read_audio(mix_path)
    summed = np.zeros_like(mix_audio, dtype=np.float32)
    blockers: list[str] = []
    for stem in STEMS:
        stem_path = stem_paths.get(stem)
        if not stem_path:
            blockers.append(f"missing_repair_stem:{stem}")
            continue
        stem_audio, stem_rate = _read_audio(Path(stem_path))
        if stem_rate != mix_rate:
            blockers.append(f"sample_rate_mismatch:{stem}:{stem_rate}!={mix_rate}")
            continue
        if stem_audio.shape != mix_audio.shape:
            blockers.append(f"shape_mismatch:{stem}:{stem_audio.shape}!={mix_audio.shape}")
            continue
        summed = summed + stem_audio
    if blockers:
        return {
            "applied": False,
            "repair_owner": "runner_conservation_repair_v1",
            "reason": "repair_inputs_invalid",
            "target_stem": target_stem,
            "blockers": blockers,
        }

    residual = mix_audio.astype(np.float32) - summed
    mix_rms = _rms(mix_audio)
    residual_rms = _rms(residual)
    if residual_rms <= 1e-12:
        return {
            "applied": False,
            "repair_owner": "runner_conservation_repair_v1",
            "reason": "residual_floor",
            "target_stem": target_stem,
            "pre_repair_residual_rms": 0.0,
            "pre_repair_mix_rms": round(mix_rms, 8),
            "pre_repair_residual_ratio": 0.0,
        }
    residual_ratio = residual_rms / max(mix_rms, 1e-12)
    if max_residual_ratio is not None and residual_ratio > max_residual_ratio:
        return {
            "applied": False,
            "repair_owner": "runner_conservation_repair_v1",
            "reason": "conservation_repair_ratio_too_high",
            "target_stem": target_stem,
            "pre_repair_residual_rms": round(residual_rms, 8),
            "pre_repair_mix_rms": round(mix_rms, 8),
            "pre_repair_residual_ratio": round(residual_ratio, 8),
            "max_residual_ratio": round(float(max_residual_ratio), 8),
        }
    target_audio, target_rate = _read_audio(Path(target_path))
    if target_rate != mix_rate or target_audio.shape != mix_audio.shape:
        return {
            "applied": False,
            "repair_owner": "runner_conservation_repair_v1",
            "reason": "target_shape_mismatch",
            "target_stem": target_stem,
        }
    _write_audio(Path(target_path), target_audio.astype(np.float32) + residual, mix_rate)
    return {
        "applied": True,
        "repair_owner": "runner_conservation_repair_v1",
        "target_stem": target_stem,
        "pre_repair_residual_rms": round(residual_rms, 8),
        "pre_repair_mix_rms": round(mix_rms, 8),
        "pre_repair_residual_ratio": round(residual_ratio, 8),
        "max_residual_ratio": round(float(max_residual_ratio), 8) if max_residual_ratio is not None else None,
    }


def apply_stem_merges(
    *,
    stem_paths: Mapping[str, str],
    merge_specs: list[Mapping[str, Any]],
) -> dict[str, Any]:
    blocker_reasons: list[str] = []
    operations: list[dict[str, str]] = []
    for spec in merge_specs:
        source_stem = str(spec.get("source_stem") or "").strip()
        target_stem = str(spec.get("target_stem") or "").strip()
        mode = str(spec.get("mode") or "move").strip()
        if source_stem not in STEMS:
            blocker_reasons.append(f"invalid_merge_source_stem:{source_stem}")
            continue
        if target_stem not in STEMS:
            blocker_reasons.append(f"invalid_merge_target_stem:{target_stem}")
            continue
        if source_stem == target_stem:
            blocker_reasons.append(f"invalid_merge_same_stem:{source_stem}")
            continue
        if mode != "move":
            blocker_reasons.append(f"unsupported_merge_mode:{mode}")
            continue
        source_path = stem_paths.get(source_stem)
        target_path = stem_paths.get(target_stem)
        if not source_path:
            blocker_reasons.append(f"missing_merge_source_path:{source_stem}")
            continue
        if not target_path:
            blocker_reasons.append(f"missing_merge_target_path:{target_stem}")
            continue
        source_audio, source_rate = _read_audio(Path(source_path))
        target_audio, target_rate = _read_audio(Path(target_path))
        if source_rate != target_rate:
            blocker_reasons.append(f"merge_sample_rate_mismatch:{source_stem}:{target_stem}:{source_rate}!={target_rate}")
            continue
        if source_audio.shape != target_audio.shape:
            blocker_reasons.append(f"merge_shape_mismatch:{source_stem}:{target_stem}:{source_audio.shape}!={target_audio.shape}")
            continue
        _write_audio(Path(target_path), target_audio.astype(np.float32) + source_audio.astype(np.float32), target_rate)
        _write_audio(Path(source_path), np.zeros_like(source_audio, dtype=np.float32), source_rate)
        operation = {
            "source_stem": source_stem,
            "target_stem": target_stem,
            "mode": mode,
            "source_path": str(source_path),
            "target_path": str(target_path),
        }
        reason = str(spec.get("reason") or "").strip()
        if reason:
            operation["reason"] = reason
        operations.append(operation)
    return {
        "applied": bool(operations),
        "repair_owner": "runner_stem_merge_v1",
        "operations": operations,
        "blocker_reasons": blocker_reasons,
    }


def apply_minus_vocal_debleed(
    *,
    stem_paths: Mapping[str, str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    low_hz = float(config.get("low_hz", 700.0))
    high_hz = float(config.get("high_hz", 1200.0))
    strength = float(config.get("strength", 1.0))
    try:
        passes = int(config.get("passes", 1))
    except (TypeError, ValueError):
        passes = 0
    target_stems_raw = config.get("target_stems")
    target_stems = (
        [str(stem) for stem in target_stems_raw]
        if isinstance(target_stems_raw, list) and target_stems_raw
        else list(NON_VOCAL_STEMS)
    )
    blockers: list[str] = []
    if not 0.0 <= strength <= 1.0:
        blockers.append("invalid_debleed_strength")
    if low_hz <= 0.0 or high_hz <= low_hz:
        blockers.append("invalid_debleed_band")
    if passes < 1 or passes > 4:
        blockers.append("invalid_debleed_passes")
    invalid_targets = [stem for stem in target_stems if stem not in DEBLEED_TARGET_STEMS]
    if invalid_targets:
        blockers.extend(f"invalid_debleed_target:{stem}" for stem in invalid_targets)
    vocals_path = stem_paths.get("vocals")
    if not vocals_path:
        blockers.append("missing_vocals_stem")
    if blockers:
        return {
            "applied": False,
            "repair_owner": "runner_minus_vocal_debleed_v1",
            "reason": "invalid_debleed_config",
            "blockers": blockers,
        }

    vocals_audio, sample_rate = _read_audio(Path(str(vocals_path)))
    stem_audios: dict[str, np.ndarray] = {}
    moved_total = np.zeros_like(vocals_audio, dtype=np.float32)
    moved_by_stem: dict[str, np.ndarray] = {}
    input_band_rms: dict[str, float] = {}
    per_stem: dict[str, dict[str, float]] = {}
    for stem in target_stems:
        stem_path = stem_paths.get(stem)
        if not stem_path:
            blockers.append(f"missing_debleed_stem:{stem}")
            continue
        stem_audio, stem_rate = _read_audio(Path(str(stem_path)))
        if stem_rate != sample_rate:
            blockers.append(f"sample_rate_mismatch:{stem}:{stem_rate}!={sample_rate}")
            continue
        if stem_audio.shape != vocals_audio.shape:
            blockers.append(f"shape_mismatch:{stem}:{stem_audio.shape}!={vocals_audio.shape}")
            continue
        stem_audios[stem] = stem_audio.astype(np.float32)
        moved_by_stem[stem] = np.zeros_like(stem_audio, dtype=np.float32)
        input_band_rms[stem] = _rms(band_limited_audio(stem_audio, sample_rate, low_hz, high_hz))
    if blockers:
        return {
            "applied": False,
            "repair_owner": "runner_minus_vocal_debleed_v1",
            "reason": "debleed_inputs_invalid",
            "blockers": blockers,
        }

    for _pass_index in range(passes):
        pass_moved_total = np.zeros_like(vocals_audio, dtype=np.float32)
        for stem, stem_audio in list(stem_audios.items()):
            moved = band_limited_audio(stem_audio, sample_rate, low_hz, high_hz) * np.float32(strength)
            stem_audios[stem] = (stem_audio.astype(np.float32) - moved).astype(np.float32)
            moved_by_stem[stem] = moved_by_stem[stem] + moved
            pass_moved_total = pass_moved_total + moved
        moved_total = moved_total + pass_moved_total
        vocals_audio = (vocals_audio.astype(np.float32) + pass_moved_total).astype(np.float32)

    moved_rms = _rms(moved_total)
    if moved_rms <= 1e-12:
        return {
            "applied": False,
            "repair_owner": "runner_minus_vocal_debleed_v1",
            "reason": "debleed_floor",
            "target_stems": target_stems,
            "moved_rms": 0.0,
            "passes": passes,
        }
    for stem, stem_audio in stem_audios.items():
        _write_audio(Path(str(stem_paths[stem])), stem_audio.astype(np.float32), sample_rate)
        before_band = float(input_band_rms.get(stem, 0.0))
        after_band = _rms(band_limited_audio(stem_audio, sample_rate, low_hz, high_hz))
        per_stem[stem] = {
            "input_band_rms": round(before_band, 8),
            "output_band_rms": round(after_band, 8),
            "moved_rms": round(_rms(moved_by_stem[stem]), 8),
            "removed_band_ratio": round(_rms(moved_by_stem[stem]) / max(before_band, 1.0e-12), 8),
            "passes": passes,
        }
    _write_audio(Path(str(vocals_path)), vocals_audio.astype(np.float32), sample_rate)
    return {
        "applied": True,
        "repair_owner": "runner_minus_vocal_debleed_v1",
        "low_hz": low_hz,
        "high_hz": high_hz,
        "strength": strength,
        "passes": passes,
        "target_stems": target_stems,
        "moved_rms": round(moved_rms, 8),
        "per_stem": per_stem,
    }


def apply_vocal_projection_debleed(
    *,
    stem_paths: Mapping[str, str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    strength = float(config.get("strength", 1.0))
    max_projection_coeff = float(config.get("max_projection_coeff", 1.5))
    min_abs_coeff = float(config.get("min_abs_coeff", 1e-4))
    target_stems_raw = config.get("target_stems")
    target_stems = (
        [str(stem) for stem in target_stems_raw]
        if isinstance(target_stems_raw, list) and target_stems_raw
        else list(NON_VOCAL_STEMS)
    )
    blockers: list[str] = []
    if not 0.0 <= strength <= 1.0:
        blockers.append("invalid_projection_debleed_strength")
    if max_projection_coeff <= 0.0:
        blockers.append("invalid_projection_debleed_max_coeff")
    if min_abs_coeff < 0.0:
        blockers.append("invalid_projection_debleed_min_coeff")
    invalid_targets = [stem for stem in target_stems if stem not in DEBLEED_TARGET_STEMS]
    if invalid_targets:
        blockers.extend(f"invalid_projection_debleed_target:{stem}" for stem in invalid_targets)
    vocals_path = stem_paths.get("vocals")
    if not vocals_path:
        blockers.append("missing_vocals_stem")
    if blockers:
        return {
            "applied": False,
            "repair_owner": "runner_vocal_projection_debleed_v1",
            "reason": "invalid_projection_debleed_config",
            "blockers": blockers,
        }

    vocals_audio, sample_rate = _read_audio(Path(str(vocals_path)))
    vocal_energy = float(np.sum(np.square(vocals_audio.astype(np.float64))))
    if vocal_energy <= 1e-12:
        return {
            "applied": False,
            "repair_owner": "runner_vocal_projection_debleed_v1",
            "reason": "vocal_projection_floor",
            "target_stems": target_stems,
            "moved_rms": 0.0,
        }

    stem_audios: dict[str, np.ndarray] = {}
    for stem in target_stems:
        stem_path = stem_paths.get(stem)
        if not stem_path:
            blockers.append(f"missing_projection_debleed_stem:{stem}")
            continue
        stem_audio, stem_rate = _read_audio(Path(str(stem_path)))
        if stem_rate != sample_rate:
            blockers.append(f"sample_rate_mismatch:{stem}:{stem_rate}!={sample_rate}")
            continue
        if stem_audio.shape != vocals_audio.shape:
            blockers.append(f"shape_mismatch:{stem}:{stem_audio.shape}!={vocals_audio.shape}")
            continue
        stem_audios[stem] = stem_audio
    if blockers:
        return {
            "applied": False,
            "repair_owner": "runner_vocal_projection_debleed_v1",
            "reason": "projection_debleed_inputs_invalid",
            "blockers": blockers,
        }

    moved_total = np.zeros_like(vocals_audio, dtype=np.float32)
    per_stem: dict[str, dict[str, float]] = {}
    for stem, stem_audio in stem_audios.items():
        projection_coeff = float(
            np.sum(stem_audio.astype(np.float64) * vocals_audio.astype(np.float64)) / vocal_energy
        )
        projection_coeff = max(0.0, min(max_projection_coeff, projection_coeff))
        if abs(projection_coeff) < min_abs_coeff:
            projection_coeff = 0.0
            moved = np.zeros_like(vocals_audio, dtype=np.float32)
        else:
            moved = (vocals_audio.astype(np.float32) * np.float32(projection_coeff * strength)).astype(np.float32)
        moved_total = moved_total + moved
        per_stem[stem] = {
            "projection_coeff": round(projection_coeff, 8),
            "moved_rms": round(_rms(moved), 8),
        }

    moved_rms = _rms(moved_total)
    if moved_rms <= 1e-12:
        return {
            "applied": False,
            "repair_owner": "runner_vocal_projection_debleed_v1",
            "reason": "projection_debleed_floor",
            "target_stems": target_stems,
            "moved_rms": 0.0,
            "per_stem": per_stem,
        }

    for stem, stem_audio in stem_audios.items():
        projection_coeff = float(per_stem[stem]["projection_coeff"])
        moved = (vocals_audio.astype(np.float32) * np.float32(projection_coeff * strength)).astype(np.float32)
        _write_audio(Path(str(stem_paths[stem])), stem_audio.astype(np.float32) - moved, sample_rate)
    _write_audio(Path(str(vocals_path)), vocals_audio.astype(np.float32) + moved_total, sample_rate)
    return {
        "applied": True,
        "repair_owner": "runner_vocal_projection_debleed_v1",
        "strength": strength,
        "max_projection_coeff": max_projection_coeff,
        "min_abs_coeff": min_abs_coeff,
        "target_stems": target_stems,
        "moved_rms": round(moved_rms, 8),
        "per_stem": per_stem,
    }


def apply_bass_stereo_width_repair(
    *,
    stem_paths: Mapping[str, str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    low_hz = float(config.get("low_hz", 20.0))
    high_hz = float(config.get("high_hz", 260.0))
    side_gain = float(config.get("side_gain", 2.5))
    max_side_gain = float(config.get("max_side_gain", 4.0))
    blockers: list[str] = []
    if low_hz <= 0.0 or high_hz <= low_hz:
        blockers.append("invalid_bass_stereo_repair_band")
    if side_gain < 1.0 or side_gain > max_side_gain:
        blockers.append("invalid_bass_stereo_repair_side_gain")
    bass_path = stem_paths.get("bass")
    if not bass_path:
        blockers.append("missing_bass_stem")
    if blockers:
        return {
            "applied": False,
            "repair_owner": "runner_bass_stereo_width_repair_v1",
            "reason": "invalid_bass_stereo_repair_config",
            "blockers": blockers,
        }

    bass_audio, sample_rate = _read_audio(Path(str(bass_path)))
    if bass_audio.ndim != 2 or bass_audio.shape[1] < 2:
        return {
            "applied": False,
            "repair_owner": "runner_bass_stereo_width_repair_v1",
            "reason": "bass_stereo_repair_requires_stereo",
            "blockers": ["bass_stem_not_stereo"],
        }
    before_width = _band_stereo_width(bass_audio, sample_rate, low_hz, high_hz)
    band = band_limited_audio(bass_audio, sample_rate, low_hz, high_hz).astype(np.float32)
    residual = bass_audio.astype(np.float32) - band
    left = band[:, 0]
    right = band[:, 1]
    mid = (left + right) * np.float32(0.5)
    side = (left - right) * np.float32(0.5)
    side_rms = _rms(side)
    if side_rms <= 1e-12:
        return {
            "applied": False,
            "repair_owner": "runner_bass_stereo_width_repair_v1",
            "reason": "bass_side_floor",
            "low_hz": low_hz,
            "high_hz": high_hz,
            "side_gain": side_gain,
            "before_width": round(before_width, 8),
        }
    widened_band = np.array(band, copy=True)
    boosted_side = side * np.float32(side_gain)
    widened_band[:, 0] = mid + boosted_side
    widened_band[:, 1] = mid - boosted_side
    repaired = residual + widened_band
    if bass_audio.shape[1] > 2:
        repaired[:, 2:] = bass_audio[:, 2:]
    _write_audio(Path(str(bass_path)), repaired.astype(np.float32), sample_rate)
    after_width = _band_stereo_width(repaired, sample_rate, low_hz, high_hz)
    return {
        "applied": True,
        "repair_owner": "runner_bass_stereo_width_repair_v1",
        "low_hz": low_hz,
        "high_hz": high_hz,
        "side_gain": side_gain,
        "max_side_gain": max_side_gain,
        "before_width": round(before_width, 8),
        "after_width": round(after_width, 8),
        "side_rms": round(side_rms, 8),
    }


def apply_vocal_depercussive_spectral_repair(
    *,
    stem_paths: Mapping[str, str],
    config: Mapping[str, Any],
    checkpoint_path: str,
) -> dict[str, Any]:
    stem = str(config.get("stem") or "vocals").strip().lower()
    source_stem = str(config.get("forensic_source_condition_stem") or config.get("source_stem") or "drums").strip().lower()
    result: dict[str, Any] = {
        "applied": False,
        "repair_owner": "runner_vocal_depercussive_spectral_repair_v1",
        "stem": stem,
        "source_stem": source_stem,
        "checkpoint": checkpoint_path,
        "blockers": [],
    }
    if stem != "vocals":
        result["blockers"].append(f"spectral_repair_invalid_stem:{stem}")
        return result
    target_path = stem_paths.get(stem)
    source_path = stem_paths.get(source_stem)
    if not target_path:
        result["blockers"].append(f"missing_spectral_repair_stem:{stem}")
        return result
    if not source_path:
        result["blockers"].append(f"missing_spectral_repair_source:{source_stem}")
        return result
    try:
        checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        if str(checkpoint.get("schema") or "") != "vocal_depercussive_spectral_repair.v1":
            result["blockers"].append("invalid_spectral_repair_checkpoint_schema")
            return result
        params = dict(checkpoint.get("parameters") or {})
        for key in (
            "strength",
            "self_hf_attenuation",
            "region_hf_attenuation",
            "hard_region_hf_suppression",
            "max_projection_gain",
            "min_region_score",
            "high_band_hz",
            "delivery_pair_compensation_strength",
            "full_region_hf_suppression",
        ):
            if key in config:
                params[key] = config[key]
        vocals, vocal_rate = _read_audio(Path(str(target_path)))
        source, source_rate = _read_audio(Path(str(source_path)))
        if vocal_rate != source_rate:
            result["blockers"].append("spectral_repair_sample_rate_mismatch")
            return result
        frames = min(vocals.shape[0], source.shape[0])
        vocals = vocals[:frames]
        source = source[:frames]
        candidate_root = Path(str(target_path)).parent.parent
        score_dir = candidate_root / "score" / "spectral_repair"
        before = score_pair_forensic(
            tested_stem=stem,
            source_stem=source_stem,
            tested_path=Path(str(target_path)),
            source_path=Path(str(source_path)),
            output_dir=score_dir / "before",
            config=None,
        )
        min_region_score = float(params.get("min_region_score", 0.55))
        regions = [
            region
            for region in before.get("regions", [])
            if float(region.get("score") or 0.0) >= min_region_score
        ]
        if not regions and before.get("regions"):
            regions = list(before.get("regions", [])[:3])
        if _parse_bool(params.get("full_region_hf_suppression", False)):
            regions = [{"start": 0.0, "end": frames / float(vocal_rate), "score": 1.0}]
        if not regions:
            result["blockers"].append("spectral_repair_no_regions")
            return result
        high_band = params.get("high_band_hz") or [1200.0, 10000.0]
        repaired, removed, applied_regions = _repair_vocal_depercussive_regions(
            vocals=vocals,
            drums=source,
            sample_rate=vocal_rate,
            regions=regions,
            strength=float(params.get("strength", 1.0)),
            self_hf_attenuation=float(params.get("self_hf_attenuation", 0.35)),
            region_hf_attenuation=float(params.get("region_hf_attenuation", 0.0)),
            hard_region_hf_suppression=_parse_bool(params.get("hard_region_hf_suppression", False)),
            max_projection_gain=float(params.get("max_projection_gain", 2.0)),
            high_band_hz=(float(high_band[0]), float(high_band[1])),
        )
        delta = (repaired.astype(np.float32) - vocals.astype(np.float32)).astype(np.float32)
        _write_audio(Path(str(target_path)), repaired.astype(np.float32), vocal_rate)
        compensation_stem = ""
        compensation_delta_rms = 0.0
        compensation_strength = max(0.0, min(1.0, float(params.get("delivery_pair_compensation_strength", 0.0))))
        if compensation_strength > 0.0 and "minus" in stem_paths:
            minus_path = Path(str(stem_paths["minus"]))
            minus_audio, minus_rate = _read_audio(minus_path)
            if minus_rate != vocal_rate or minus_audio.shape[0] < frames or minus_audio.shape[1] != repaired.shape[1]:
                result["blockers"].append("spectral_repair_minus_compensation_shape_mismatch")
                return result
            compensation = (-delta * np.float32(compensation_strength)).astype(np.float32)
            minus_repaired = np.clip(minus_audio[:frames].astype(np.float32) + compensation, -1.0, 1.0).astype(np.float32)
            if minus_audio.shape[0] > frames:
                merged_minus = minus_audio.astype(np.float32).copy()
                merged_minus[:frames] = minus_repaired
                minus_repaired = merged_minus
            _write_audio(minus_path, minus_repaired, minus_rate)
            compensation_stem = "minus"
            compensation_delta_rms = _rms(compensation)
        after = score_pair_forensic(
            tested_stem=stem,
            source_stem=source_stem,
            tested_path=Path(str(target_path)),
            source_path=Path(str(source_path)),
            output_dir=score_dir / "after",
            config=None,
        )
        result.update(
            {
                "applied": True,
                "parameters": params,
                "region_count": len(applied_regions),
                "applied_regions": applied_regions,
                "removed_rms": round(_rms(removed), 8),
                "delta_rms": round(_rms(delta), 8),
                "forensic_before_verdict": before.get("verdict"),
                "forensic_after_verdict": after.get("verdict"),
                "forensic_after_blockers": after.get("blocker_reasons") or [],
                "delivery_pair_compensation_stem": compensation_stem,
                "delivery_pair_compensation_delta_rms": round(compensation_delta_rms, 8),
                "diagnostics": {
                    "target_issue_improved": before.get("verdict") != "pass" and after.get("verdict") == "pass",
                    "before_imprint_score": before.get("imprint_score"),
                    "after_imprint_score": after.get("imprint_score"),
                },
            }
        )
    except Exception as exc:  # pragma: no cover - defensive manifest evidence
        result["blockers"].append(f"spectral_repair_failed:{type(exc).__name__}")
        result["error"] = str(exc)
    return result


def apply_learned_stem_region_repair(
    *,
    mix_path: Path,
    stem_paths: Mapping[str, str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    stem = str(config.get("stem") or "bass").strip().lower()
    issue_type = str(config.get("issue_type") or ISSUE_STEREO_COLLAPSE).strip().lower()
    checkpoint_path = str(config.get("checkpoint") or config.get("checkpoint_path") or "").strip()
    result: dict[str, Any] = {
        "applied": False,
        "repair_owner": "runner_stem_region_repair_v1",
        "stem": stem,
        "issue_type": issue_type,
        "checkpoint": checkpoint_path,
        "blockers": [],
    }
    if not checkpoint_path:
        result["blockers"].append("missing_stem_region_repair_checkpoint")
        return result
    target_path = stem_paths.get(stem)
    if not target_path:
        result["blockers"].append(f"missing_repair_stem:{stem}")
        return result
    try:
        from tools.stem_region_repair_model import apply_region_repair_model, load_region_repair_checkpoint

        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.suffix.lower() == ".json":
            checkpoint_probe = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            if str(checkpoint_probe.get("schema") or "") == "vocal_depercussive_spectral_repair.v1":
                return apply_vocal_depercussive_spectral_repair(
                    stem_paths=stem_paths,
                    config=config,
                    checkpoint_path=checkpoint_path,
                )
        model, checkpoint = load_region_repair_checkpoint(Path(checkpoint_path))
        training_history = checkpoint.get("training_history") if isinstance(checkpoint.get("training_history"), Mapping) else {}
        checkpoint_issue = str(training_history.get("issue_type") or "").strip().lower()
        if checkpoint_issue and checkpoint_issue != issue_type:
            result["blockers"].append(f"stem_region_repair_issue_mismatch:{checkpoint_issue}:{issue_type}")
            return result
        mix_audio, mix_rate = _read_audio(mix_path)
        bad_audio, stem_rate = _read_audio(Path(str(target_path)))
        if mix_rate != stem_rate:
            result["blockers"].append("stem_region_repair_sample_rate_mismatch")
            return result
        frames = min(mix_audio.shape[0], bad_audio.shape[0])
        mix_audio = mix_audio[:frames]
        bad_audio = bad_audio[:frames]
        source_condition_stem = str(
            config.get("forensic_source_condition_stem")
            or training_history.get("forensic_source_condition_stem")
            or ""
        ).strip().lower()
        source_condition_audio: np.ndarray | None = None
        source_condition_path = ""
        if int(getattr(model, "input_channels", 6)) > 6 or source_condition_stem:
            if not source_condition_stem:
                result["blockers"].append("missing_forensic_source_condition_stem")
                return result
            source_condition_path = stem_paths.get(source_condition_stem) or ""
            if not source_condition_path:
                result["blockers"].append(f"missing_forensic_source_condition_audio:{source_condition_stem}")
                return result
            source_condition_audio, source_condition_rate = _read_audio(Path(str(source_condition_path)))
            if source_condition_rate != stem_rate:
                result["blockers"].append("stem_region_repair_source_condition_sample_rate_mismatch")
                return result
            if source_condition_audio.shape[0] < frames or source_condition_audio.shape[1] != bad_audio.shape[1]:
                result["blockers"].append("stem_region_repair_source_condition_shape_mismatch")
                return result
            source_condition_audio = source_condition_audio[:frames]
        region = config.get("region") if isinstance(config.get("region"), Mapping) else training_history.get("region")
        if not isinstance(region, Mapping):
            region = {"start_sec": 0.0, "end_sec": frames / float(mix_rate)}
        start_sec = float(region.get("start_sec", 0.0))
        end_sec = float(region.get("end_sec", frames / float(mix_rate)))
        mask = make_region_mask(frame_count=frames, sample_rate=mix_rate, start_sec=start_sec, end_sec=end_sec)
        if float(np.max(mask)) <= 0.0:
            result["blockers"].append("stem_region_repair_empty_region")
            result["region"] = {"start_sec": start_sec, "end_sec": end_sec}
            return result
        before_width = _stereo_width(bad_audio)
        repaired, diagnostics = apply_region_repair_model(
            model=model,
            mix_audio=mix_audio,
            bad_stem_audio=bad_audio,
            region_mask=mask,
            sample_rate=mix_rate,
            issue_type=issue_type,
            source_condition_audio=source_condition_audio,
        )
        if stem == "bass" and issue_type == ISSUE_STEREO_COLLAPSE and bool(config.get("bass_band_delta_only", True)):
            low_hz = float(config.get("low_hz", 20.0))
            high_hz = float(config.get("high_hz", 260.0))
            delta = band_limited_audio((repaired - bad_audio).astype(np.float32), mix_rate, low_hz, high_hz).astype(np.float32)
            repaired = np.clip(bad_audio.astype(np.float32) + delta, -1.0, 1.0).astype(np.float32)
            diagnostics["bass_band_delta_only"] = True
            diagnostics["bass_band_delta_hz"] = [low_hz, high_hz]
        companion_stem = str(
            config.get("mix_preserve_companion_stem") or training_history.get("mix_preserve_companion_stem") or ""
        ).strip().lower()
        companion_adjusted: np.ndarray | None = None
        companion_path: str | None = None
        delivery_pair_compensation_stem = ""
        delivery_pair_compensation_delta: np.ndarray | None = None
        delivery_pair_compensation_path: str | None = None
        delivery_pair_compensation_audio: np.ndarray | None = None
        compensation_delta = (repaired.astype(np.float32) - bad_audio.astype(np.float32)).astype(np.float32)
        raw_delivery_pair_compensation_enabled = config.get(
            "delivery_pair_compensation_enabled",
            training_history.get("delivery_pair_compensation_enabled"),
        )
        delivery_pair_compensation_enabled = (
            issue_type != ISSUE_BAD_RECONSTRUCTION
            if raw_delivery_pair_compensation_enabled is None
            else _parse_bool(raw_delivery_pair_compensation_enabled)
        )
        if companion_stem:
            if companion_stem == stem:
                result["blockers"].append(f"stem_region_repair_invalid_companion:{companion_stem}")
                return result
            companion_path = stem_paths.get(companion_stem)
            if not companion_path:
                result["blockers"].append(f"missing_repair_companion_stem:{companion_stem}")
                return result
            companion_audio, companion_rate = _read_audio(Path(str(companion_path)))
            if companion_rate != stem_rate:
                result["blockers"].append("stem_region_repair_companion_sample_rate_mismatch")
                return result
            if companion_audio.shape != bad_audio.shape:
                result["blockers"].append("stem_region_repair_companion_shape_mismatch")
                return result
            companion_adjusted = np.clip(
                companion_audio.astype(np.float32) - compensation_delta,
                -1.0,
                1.0,
            ).astype(np.float32)
        vocal_pair_delta = np.zeros_like(compensation_delta, dtype=np.float32)
        if delivery_pair_compensation_enabled and stem == "vocals" and companion_stem != "minus":
            if issue_type == ISSUE_INSTRUMENT_LEAK and source_condition_stem:
                forensic_band = training_history.get("forensic_high_band_hz")
                if isinstance(forensic_band, (str, bytes)) or not isinstance(forensic_band, Sequence) or len(forensic_band) != 2:
                    forensic_band = config.get("forensic_high_band_hz")
                low_hz, high_hz = (
                    [float(value) for value in forensic_band]
                    if not isinstance(forensic_band, (str, bytes)) and isinstance(forensic_band, Sequence) and len(forensic_band) == 2
                    else [1200.0, 10000.0]
                )
                band_strength = float(
                    config.get(
                        "delivery_pair_compensation_strength",
                        training_history.get("delivery_pair_compensation_strength", 0.35),
                    )
                )
                band_strength = max(0.0, min(1.0, band_strength))
                vocal_pair_delta += (
                    band_limited_audio(compensation_delta, mix_rate, low_hz, high_hz).astype(np.float32)
                    * np.float32(band_strength)
                )
                diagnostics["delivery_pair_compensation_band_hz"] = [low_hz, high_hz]
                diagnostics["delivery_pair_compensation_strength"] = band_strength
            elif issue_type != ISSUE_INSTRUMENT_LEAK:
                vocal_pair_delta += compensation_delta
        if companion_stem == "vocals":
            vocal_pair_delta -= compensation_delta
        if stem != "minus" and _rms(vocal_pair_delta) > 1.0e-9:
            delivery_pair_compensation_stem = "minus"
            delivery_pair_compensation_path = stem_paths.get(delivery_pair_compensation_stem)
            if not delivery_pair_compensation_path:
                result["blockers"].append("missing_delivery_pair_compensation_stem:minus")
                return result
            minus_audio, minus_rate = _read_audio(Path(str(delivery_pair_compensation_path)))
            if minus_rate != stem_rate:
                result["blockers"].append("stem_region_repair_delivery_pair_sample_rate_mismatch")
                return result
            if minus_audio.shape != bad_audio.shape:
                result["blockers"].append("stem_region_repair_delivery_pair_shape_mismatch")
                return result
            delivery_pair_compensation_delta = (-vocal_pair_delta).astype(np.float32)
            delivery_pair_compensation_audio = np.clip(
                minus_audio.astype(np.float32) + delivery_pair_compensation_delta,
                -1.0,
                1.0,
            ).astype(np.float32)
        after_width = _stereo_width(repaired)
        before_band_width = _band_stereo_width(bad_audio, mix_rate, 20.0, 260.0) if stem == "bass" else before_width
        after_band_width = _band_stereo_width(repaired, mix_rate, 20.0, 260.0) if stem == "bass" else after_width
        delivery_pair_residual_before: float | None = None
        delivery_pair_residual_after: float | None = None
        if issue_type == ISSUE_BAD_RECONSTRUCTION and stem in {"vocals", "minus"}:
            paired_stem = "minus" if stem == "vocals" else "vocals"
            paired_path = stem_paths.get(paired_stem)
            if paired_path:
                paired_audio, paired_rate = _read_audio(Path(str(paired_path)))
                if paired_rate == stem_rate:
                    pair_frames = min(mix_audio.shape[0], bad_audio.shape[0], repaired.shape[0], paired_audio.shape[0])
                    if pair_frames > 0:
                        paired_after = paired_audio[:pair_frames]
                        if (
                            paired_stem == delivery_pair_compensation_stem
                            and delivery_pair_compensation_audio is not None
                        ):
                            paired_after = delivery_pair_compensation_audio[:pair_frames]
                        if stem == "vocals":
                            before_pair_sum = bad_audio[:pair_frames] + paired_audio[:pair_frames]
                            after_pair_sum = repaired[:pair_frames] + paired_after
                        else:
                            before_pair_sum = paired_audio[:pair_frames] + bad_audio[:pair_frames]
                            after_pair_sum = paired_after + repaired[:pair_frames]
                        mix_pair = mix_audio[:pair_frames]
                        delivery_pair_residual_before = _rms((mix_pair - before_pair_sum).astype(np.float32))
                        delivery_pair_residual_after = _rms((mix_pair - after_pair_sum).astype(np.float32))
        _write_audio(Path(str(target_path)), repaired.astype(np.float32), stem_rate)
        if companion_adjusted is not None and companion_path is not None:
            _write_audio(Path(str(companion_path)), companion_adjusted, stem_rate)
        if delivery_pair_compensation_delta is not None and delivery_pair_compensation_path is not None:
            _write_audio(Path(str(delivery_pair_compensation_path)), delivery_pair_compensation_audio, stem_rate)
        result.update(
            {
                "applied": True,
                "before_width": round(before_width, 8),
                "after_width": round(after_width, 8),
                "before_band_width": round(before_band_width, 8),
                "after_band_width": round(after_band_width, 8),
                "region": {"start_sec": start_sec, "end_sec": end_sec},
                "mix_preserve_companion_stem": companion_stem,
                "forensic_source_condition_stem": source_condition_stem,
                "forensic_source_condition_path": source_condition_path,
                "mix_preserve_delta_rms": round(_rms(compensation_delta), 8),
                "delivery_pair_compensation_enabled": delivery_pair_compensation_enabled,
                "delivery_pair_compensation_stem": delivery_pair_compensation_stem,
                "delivery_pair_compensation_delta_rms": (
                    round(_rms(delivery_pair_compensation_delta), 8)
                    if delivery_pair_compensation_delta is not None
                    else 0.0
                ),
                "diagnostics": {
                    **diagnostics,
                    "stereo_width_before": before_width,
                    "stereo_width_after": after_width,
                    "bass_band_stereo_width_before": before_band_width,
                    "bass_band_stereo_width_after": after_band_width,
                    **(
                        {
                            "delivery_pair_residual_before": delivery_pair_residual_before,
                            "delivery_pair_residual_after": delivery_pair_residual_after,
                        }
                        if delivery_pair_residual_before is not None and delivery_pair_residual_after is not None
                        else {}
                    ),
                    "target_issue_improved": (
                        delivery_pair_residual_after < delivery_pair_residual_before
                        if (
                            issue_type == ISSUE_BAD_RECONSTRUCTION
                            and delivery_pair_residual_before is not None
                            and delivery_pair_residual_after is not None
                        )
                        else
                        after_band_width > before_band_width if stem == "bass" and issue_type == ISSUE_STEREO_COLLAPSE
                        else after_width > before_width if issue_type == ISSUE_STEREO_COLLAPSE
                        else False
                    ),
                },
            }
        )
    except Exception as exc:  # pragma: no cover - defensive manifest evidence
        result["blockers"].append(f"stem_region_repair_failed:{type(exc).__name__}")
        result["error"] = str(exc)
    return result


def apply_learned_stem_region_repairs(
    *,
    mix_path: Path,
    stem_paths: Mapping[str, str],
    configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "applied": False,
        "repair_owner": "runner_stem_region_repair_sequence_v1",
        "repairs": [],
        "blockers": [],
    }
    if not configs:
        result["blockers"].append("missing_stem_region_repair_sequence")
        return result

    for raw_config in configs:
        if not isinstance(raw_config, Mapping):
            result["blockers"].append("invalid_stem_region_repair_sequence_item")
            continue
        item = apply_learned_stem_region_repair(
            mix_path=mix_path,
            stem_paths=stem_paths,
            config=dict(raw_config),
        )
        result["repairs"].append(item)
        if item.get("applied"):
            result["applied"] = True
        for blocker in item.get("blockers") or []:
            result["blockers"].append(str(blocker))
    return result


def _stem_region_repair_items(stem_region_repair: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(stem_region_repair, Mapping):
        return []
    sequence = stem_region_repair.get("repairs")
    if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes)):
        return [item for item in sequence if isinstance(item, Mapping)]
    return [stem_region_repair]


def _delivery_pair_residual_blocker_after_stem_region_repair(
    stem_region_repair: Mapping[str, Any] | None,
    residual_config: Mapping[str, Any],
) -> str:
    target_stem = str(residual_config.get("target_stem") or "minus").strip().lower()
    if target_stem != "minus":
        return ""
    for item in _stem_region_repair_items(stem_region_repair):
        if not item.get("applied"):
            continue
        stem = str(item.get("stem") or "").strip().lower()
        issue_type = str(item.get("issue_type") or "").strip().lower()
        if stem == "vocals" and issue_type == ISSUE_INSTRUMENT_LEAK:
            return "delivery_pair_residual_after_vocal_instrument_leak_repair_blocked"
    return ""


def apply_delivery_pair_gain_repair(
    *,
    mix_path: Path,
    stem_paths: Mapping[str, str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    target_stem = str(config.get("target_stem") or "minus")
    max_abs_gain_delta = float(config.get("max_abs_gain_delta", 0.08))
    min_abs_gain_delta = float(config.get("min_abs_gain_delta", 1e-4))
    blockers: list[str] = []
    if target_stem not in {"minus", "vocals"}:
        blockers.append(f"invalid_delivery_pair_gain_target:{target_stem}")
    if max_abs_gain_delta < 0.0 or max_abs_gain_delta > 0.25:
        blockers.append("invalid_delivery_pair_gain_delta")
    mix_audio, mix_rate = _read_audio(mix_path)
    vocals_path = stem_paths.get("vocals")
    minus_path = stem_paths.get("minus")
    target_path = stem_paths.get(target_stem)
    if not vocals_path:
        blockers.append("missing_delivery_pair_vocals")
    if not minus_path:
        blockers.append("missing_delivery_pair_minus")
    if not target_path:
        blockers.append(f"missing_delivery_pair_target:{target_stem}")
    if blockers:
        return {
            "applied": False,
            "repair_owner": "runner_delivery_pair_gain_repair_v1",
            "reason": "invalid_delivery_pair_gain_config",
            "blockers": blockers,
        }
    vocals_audio, vocals_rate = _read_audio(Path(str(vocals_path)))
    minus_audio, minus_rate = _read_audio(Path(str(minus_path)))
    if vocals_rate != mix_rate or minus_rate != mix_rate:
        return {
            "applied": False,
            "repair_owner": "runner_delivery_pair_gain_repair_v1",
            "reason": "delivery_pair_gain_sample_rate_mismatch",
            "blockers": ["delivery_pair_gain_sample_rate_mismatch"],
        }
    if vocals_audio.shape != mix_audio.shape or minus_audio.shape != mix_audio.shape:
        return {
            "applied": False,
            "repair_owner": "runner_delivery_pair_gain_repair_v1",
            "reason": "delivery_pair_gain_shape_mismatch",
            "blockers": ["delivery_pair_gain_shape_mismatch"],
        }
    target_audio = minus_audio if target_stem == "minus" else vocals_audio
    fixed_audio = vocals_audio if target_stem == "minus" else minus_audio
    desired = mix_audio.astype(np.float64) - fixed_audio.astype(np.float64)
    target = target_audio.astype(np.float64)
    denom = float(np.sum(target * target))
    if denom <= 1e-18:
        return {
            "applied": False,
            "repair_owner": "runner_delivery_pair_gain_repair_v1",
            "reason": "delivery_pair_gain_target_floor",
            "target_stem": target_stem,
        }
    optimal_gain = float(np.sum(target * desired) / denom)
    gain = max(1.0 - max_abs_gain_delta, min(1.0 + max_abs_gain_delta, optimal_gain))
    gain_delta = gain - 1.0
    before_residual = mix_audio.astype(np.float32) - vocals_audio.astype(np.float32) - minus_audio.astype(np.float32)
    before_ratio = _rms(before_residual) / max(_rms(mix_audio), 1e-12)
    if abs(gain_delta) < min_abs_gain_delta:
        return {
            "applied": False,
            "repair_owner": "runner_delivery_pair_gain_repair_v1",
            "reason": "delivery_pair_gain_floor",
            "target_stem": target_stem,
            "optimal_gain": round(optimal_gain, 8),
            "gain": round(gain, 8),
            "before_residual_ratio": round(before_ratio, 8),
        }
    repaired_target = (target_audio.astype(np.float32) * np.float32(gain)).astype(np.float32)
    if target_stem == "minus":
        repaired_vocals = vocals_audio.astype(np.float32)
        repaired_minus = repaired_target
    else:
        repaired_vocals = repaired_target
        repaired_minus = minus_audio.astype(np.float32)
    after_residual = mix_audio.astype(np.float32) - repaired_vocals - repaired_minus
    after_ratio = _rms(after_residual) / max(_rms(mix_audio), 1e-12)
    if after_ratio >= before_ratio:
        return {
            "applied": False,
            "repair_owner": "runner_delivery_pair_gain_repair_v1",
            "reason": "delivery_pair_gain_no_improvement",
            "target_stem": target_stem,
            "optimal_gain": round(optimal_gain, 8),
            "gain": round(gain, 8),
            "before_residual_ratio": round(before_ratio, 8),
            "after_residual_ratio": round(after_ratio, 8),
        }
    _write_audio(Path(str(target_path)), repaired_target, mix_rate)
    return {
        "applied": True,
        "repair_owner": "runner_delivery_pair_gain_repair_v1",
        "target_stem": target_stem,
        "optimal_gain": round(optimal_gain, 8),
        "gain": round(gain, 8),
        "gain_delta": round(gain_delta, 8),
        "max_abs_gain_delta": max_abs_gain_delta,
        "before_residual_ratio": round(before_ratio, 8),
        "after_residual_ratio": round(after_ratio, 8),
    }


def apply_delivery_pair_residual_repair(
    *,
    mix_path: Path,
    stem_paths: Mapping[str, str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    target_stem = str(config.get("target_stem") or "minus")
    max_residual_ratio = float(config.get("max_residual_ratio", 0.08))
    blockers: list[str] = []
    if target_stem not in {"minus", "vocals"}:
        blockers.append(f"invalid_delivery_pair_residual_target:{target_stem}")
    if max_residual_ratio < 0.0 or max_residual_ratio > 0.25:
        blockers.append("invalid_delivery_pair_residual_ratio")
    mix_audio, mix_rate = _read_audio(mix_path)
    vocals_path = stem_paths.get("vocals")
    minus_path = stem_paths.get("minus")
    target_path = stem_paths.get(target_stem)
    if not vocals_path:
        blockers.append("missing_delivery_pair_vocals")
    if not minus_path:
        blockers.append("missing_delivery_pair_minus")
    if not target_path:
        blockers.append(f"missing_delivery_pair_target:{target_stem}")
    if blockers:
        return {
            "applied": False,
            "repair_owner": "runner_delivery_pair_residual_repair_v1",
            "reason": "invalid_delivery_pair_residual_config",
            "blockers": blockers,
        }
    vocals_audio, vocals_rate = _read_audio(Path(str(vocals_path)))
    minus_audio, minus_rate = _read_audio(Path(str(minus_path)))
    if vocals_rate != mix_rate or minus_rate != mix_rate:
        return {
            "applied": False,
            "repair_owner": "runner_delivery_pair_residual_repair_v1",
            "reason": "delivery_pair_residual_sample_rate_mismatch",
            "blockers": ["delivery_pair_residual_sample_rate_mismatch"],
        }
    if vocals_audio.shape != mix_audio.shape or minus_audio.shape != mix_audio.shape:
        return {
            "applied": False,
            "repair_owner": "runner_delivery_pair_residual_repair_v1",
            "reason": "delivery_pair_residual_shape_mismatch",
            "blockers": ["delivery_pair_residual_shape_mismatch"],
        }
    residual = mix_audio.astype(np.float32) - vocals_audio.astype(np.float32) - minus_audio.astype(np.float32)
    before_ratio = _rms(residual) / max(_rms(mix_audio), 1e-12)
    if before_ratio <= 1e-12:
        return {
            "applied": False,
            "repair_owner": "runner_delivery_pair_residual_repair_v1",
            "reason": "delivery_pair_residual_floor",
            "target_stem": target_stem,
            "before_residual_ratio": 0.0,
        }
    if before_ratio > max_residual_ratio:
        return {
            "applied": False,
            "repair_owner": "runner_delivery_pair_residual_repair_v1",
            "reason": "delivery_pair_residual_ratio_too_high",
            "target_stem": target_stem,
            "before_residual_ratio": round(before_ratio, 8),
            "max_residual_ratio": round(max_residual_ratio, 8),
        }
    target_audio = minus_audio if target_stem == "minus" else vocals_audio
    repaired_target = (target_audio.astype(np.float32) + residual).astype(np.float32)
    if target_stem == "minus":
        repaired_vocals = vocals_audio.astype(np.float32)
        repaired_minus = repaired_target
    else:
        repaired_vocals = repaired_target
        repaired_minus = minus_audio.astype(np.float32)
    after_residual = mix_audio.astype(np.float32) - repaired_vocals - repaired_minus
    after_ratio = _rms(after_residual) / max(_rms(mix_audio), 1e-12)
    if after_ratio >= before_ratio:
        return {
            "applied": False,
            "repair_owner": "runner_delivery_pair_residual_repair_v1",
            "reason": "delivery_pair_residual_no_improvement",
            "target_stem": target_stem,
            "before_residual_ratio": round(before_ratio, 8),
            "after_residual_ratio": round(after_ratio, 8),
        }
    _write_audio(Path(str(target_path)), repaired_target, mix_rate)
    return {
        "applied": True,
        "repair_owner": "runner_delivery_pair_residual_repair_v1",
        "target_stem": target_stem,
        "before_residual_ratio": round(before_ratio, 8),
        "after_residual_ratio": round(after_ratio, 8),
        "max_residual_ratio": round(max_residual_ratio, 8),
    }


def _find_window_candidate(window: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    for candidate in window.get("candidates", []):
        if str(candidate.get("candidate_id") or "") == candidate_id and candidate.get("verdict") == "auto_accept":
            return candidate
    raise ValueError(f"accepted candidate not found for window {window.get('window_id')}: {candidate_id}")


def _config_identity(config: Mapping[str, Any]) -> str:
    return json.dumps(dict(config), sort_keys=True, separators=(",", ":"), default=str)


def _resolve_listen_pack_forensic_config(
    *,
    recipe: Mapping[str, Any],
    windows: Sequence[Mapping[str, Any]],
    accepted_window_candidates: Mapping[str, str],
) -> tuple[dict[str, Any], list[str]]:
    recipe_config = _cross_stem_forensic_gate_config(recipe)
    accepted_configs: list[dict[str, Any]] = []
    for window in windows:
        window_id = str(window.get("window_id") or "")
        candidate_id = str(accepted_window_candidates.get(window_id) or "")
        if not candidate_id:
            continue
        candidate = _find_window_candidate(window, candidate_id)
        candidate_config = _cross_stem_forensic_gate_config(candidate)
        if candidate_config:
            accepted_configs.append(candidate_config)
    if not accepted_configs:
        return recipe_config, []
    first = accepted_configs[0]
    first_key = _config_identity(first)
    if any(_config_identity(config) != first_key for config in accepted_configs[1:]):
        return {}, ["listen_pack_forensic_config_conflict"]
    return first, []


def _concat_audio(paths: list[Path]) -> tuple[np.ndarray, int]:
    if not paths:
        raise ValueError("cannot concatenate empty audio path list")
    chunks: list[np.ndarray] = []
    sample_rate: int | None = None
    channels: tuple[int, ...] | None = None
    for path in paths:
        audio, rate = _read_audio(path)
        if sample_rate is None:
            sample_rate = rate
            channels = audio.shape[1:]
        elif rate != sample_rate:
            raise ValueError(f"sample rate mismatch in listen pack: {path}")
        if audio.shape[1:] != channels:
            raise ValueError(f"channel shape mismatch in listen pack: {path}")
        chunks.append(audio)
    return np.concatenate(chunks, axis=0).astype(np.float32), int(sample_rate)


def _profile_name(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("profile") or candidate.get("model_split_profile") or "").strip()


def _is_model_split_v2(candidate: Mapping[str, Any]) -> bool:
    return _profile_name(candidate) == MODEL_SPLIT_V2_PROFILE


def _candidate_minus_source(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("minus_source") or "minus_lane")


def _stereo_width(audio: np.ndarray) -> float:
    if audio.ndim < 2 or audio.shape[1] < 2:
        return 0.0
    left = audio[:, 0].astype(np.float64)
    right = audio[:, 1].astype(np.float64)
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    return float(_rms(side) / max(_rms(mid), 1e-12))


def _band_stereo_width(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    if audio.ndim < 2 or audio.shape[1] < 2:
        return 0.0
    band_audio = band_limited_audio(audio, sample_rate, low_hz, high_hz)
    return _stereo_width(band_audio)


def _aligned_flattened_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    frames = min(len(a), len(b))
    channels = min(a.shape[1], b.shape[1])
    if frames <= 0 or channels <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    return (
        a[:frames, :channels].astype(np.float64).reshape(-1),
        b[:frames, :channels].astype(np.float64).reshape(-1),
    )


def _projection_ratio(stem_audio: np.ndarray, reference_audio: np.ndarray) -> tuple[float, float]:
    stem, reference = _aligned_flattened_pair(stem_audio, reference_audio)
    if stem.size == 0 or reference.size == 0:
        return 0.0, 0.0
    denom = float(np.dot(reference, reference))
    if denom <= 1e-18:
        return 0.0, 0.0
    coeff = float(np.dot(stem, reference) / denom)
    projected = coeff * reference
    ratio = float(_rms(projected) / max(_rms(stem), 1e-12))
    return abs(coeff), ratio


def score_model_split_v2_damage_gates(
    *,
    candidate: Mapping[str, Any],
    mix_path: Path,
    stem_paths: Mapping[str, str],
    stem_merges: Mapping[str, Any] | None = None,
    minus_vocal_debleed: Mapping[str, Any] | None = None,
    post_minus_vocal_debleed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = candidate.get("damage_gates") if isinstance(candidate.get("damage_gates"), dict) else {}
    enabled = _is_model_split_v2(candidate) or bool(config.get("enabled", False))
    if config.get("enabled") is False and _is_model_split_v2(candidate):
        return {
            "verdict": "blocked_damage",
            "enabled": False,
            "blocker_reasons": ["v2_damage_gate_disabled"],
            "metrics": {},
        }
    if config.get("enabled") is False:
        return {"verdict": "auto_accept", "enabled": False, "blocker_reasons": [], "metrics": {}}
    if not enabled:
        return {"verdict": "auto_accept", "enabled": False, "blocker_reasons": [], "metrics": {}}

    blockers: list[str] = []
    metrics: dict[str, Any] = {}
    if stem_merges and (stem_merges.get("applied") or candidate.get("stem_merges")):
        blockers.append("v2_repair_route_not_publishable:stem_merges")
    if minus_vocal_debleed and (minus_vocal_debleed.get("applied") or candidate.get("minus_vocal_debleed")):
        blockers.append("v2_repair_route_not_publishable:minus_vocal_debleed")
    if post_minus_vocal_debleed and (post_minus_vocal_debleed.get("applied") or candidate.get("post_minus_vocal_debleed")):
        blockers.append("v2_repair_route_not_publishable:post_minus_vocal_debleed")

    min_non_vocal_rms = float(config.get("min_non_vocal_rms", 1e-5))
    min_relative_stereo_width = float(config.get("min_relative_stereo_width", 0.20))
    min_mix_stereo_width = float(config.get("min_mix_stereo_width", 0.02))
    bass_reference_low_hz = float(config.get("bass_stereo_reference_low_hz", 20.0))
    bass_reference_high_hz = float(config.get("bass_stereo_reference_high_hz", 260.0))
    mix_audio, mix_rate = _read_audio(mix_path)
    mix_width = _stereo_width(mix_audio)
    bass_mix_width = _band_stereo_width(mix_audio, mix_rate, bass_reference_low_hz, bass_reference_high_hz)
    metrics["mix"] = {
        "rms": round(_rms(mix_audio), 8),
        "stereo_width": round(mix_width, 8),
        "bass_band_stereo_width": round(bass_mix_width, 8),
        "bass_stereo_reference_band_hz": [bass_reference_low_hz, bass_reference_high_hz],
    }
    for stem in [*NON_VOCAL_STEMS, "minus"]:
        stem_path = stem_paths.get(stem)
        if not stem_path:
            if stem == "minus" and _candidate_minus_source(candidate) == "minus_lane":
                blockers.append("missing_minus_lane_output")
            continue
        stem_audio, stem_rate = _read_audio(Path(stem_path))
        stem_rms = _rms(stem_audio)
        stem_width = _stereo_width(stem_audio)
        reference_width = mix_width
        reference_surface = "full_band_mix"
        metrics[stem] = {
            "rms": round(stem_rms, 8),
            "stereo_width": round(stem_width, 8),
            "sample_rate": stem_rate,
            "stereo_reference_surface": reference_surface,
            "stereo_reference_width": round(reference_width, 8),
        }
        if stem == "bass" and stem_rate == mix_rate:
            stem_width = _band_stereo_width(stem_audio, stem_rate, bass_reference_low_hz, bass_reference_high_hz)
            reference_width = bass_mix_width
            reference_surface = "bass_band_mix"
            metrics[stem].update(
                {
                    "bass_band_stereo_width": round(stem_width, 8),
                    "stereo_reference_surface": reference_surface,
                    "stereo_reference_width": round(reference_width, 8),
                    "bass_stereo_reference_band_hz": [bass_reference_low_hz, bass_reference_high_hz],
                }
            )
        if stem in NON_VOCAL_STEMS and stem_rms < min_non_vocal_rms:
            blockers.append(f"stem_detail_floor:{stem}")
        if (
            stem_rate == mix_rate
            and reference_width >= min_mix_stereo_width
            and stem_rms >= max(min_non_vocal_rms, 1e-12)
            and stem_width < reference_width * min_relative_stereo_width
        ):
            blockers.append(f"stereo_width_collapse:{stem}")

    return {
        "verdict": "blocked_damage" if blockers else "auto_accept",
        "enabled": True,
        "blocker_reasons": blockers,
        "metrics": metrics,
        "gate_config": {
            "min_non_vocal_rms": min_non_vocal_rms,
            "min_relative_stereo_width": min_relative_stereo_width,
            "min_mix_stereo_width": min_mix_stereo_width,
            "bass_stereo_reference_low_hz": bass_reference_low_hz,
            "bass_stereo_reference_high_hz": bass_reference_high_hz,
        },
    }


def score_model_split_v2_voice_presence_gate(
    *,
    candidate: Mapping[str, Any],
    stem_paths: Mapping[str, str],
) -> dict[str, Any]:
    config = candidate.get("voice_presence_gate") if isinstance(candidate.get("voice_presence_gate"), dict) else {}
    enabled = _is_model_split_v2(candidate) or bool(config.get("enabled", False))
    if config.get("enabled") is False and _is_model_split_v2(candidate):
        return {
            "verdict": "blocked_unknown",
            "enabled": False,
            "blocker_reasons": ["v2_voice_presence_gate_disabled"],
            "metrics": {},
        }
    if config.get("enabled") is False:
        return {"verdict": "auto_accept", "enabled": False, "blocker_reasons": [], "metrics": {}}
    if not enabled:
        return {"verdict": "auto_accept", "enabled": False, "blocker_reasons": [], "metrics": {}}
    vocals_path = stem_paths.get("vocals")
    if not vocals_path:
        return {
            "verdict": "blocked_unknown",
            "enabled": True,
            "blocker_reasons": ["missing_voice_presence_reference:vocals"],
            "metrics": {},
        }

    vocals_audio, vocals_rate = _read_audio(Path(vocals_path))
    max_projection_ratio = float(config.get("max_projection_ratio", 0.12))
    max_projection_coeff = float(config.get("max_projection_coeff", 0.18))
    blockers: list[str] = []
    metrics: dict[str, Any] = {}
    for stem in [*NON_VOCAL_STEMS, "minus"]:
        stem_path = stem_paths.get(stem)
        if not stem_path:
            continue
        stem_audio, stem_rate = _read_audio(Path(stem_path))
        if stem_rate != vocals_rate:
            blockers.append(f"voice_presence_sample_rate_mismatch:{stem}")
            continue
        coeff, ratio = _projection_ratio(stem_audio, vocals_audio)
        metrics[stem] = {
            "projection_coeff": round(coeff, 8),
            "projection_ratio": round(ratio, 8),
        }
        if coeff > max_projection_coeff and ratio > max_projection_ratio:
            blockers.append(f"voice_presence_fullband:{stem}")
    return {
        "verdict": "blocked_leak" if blockers else "auto_accept",
        "enabled": True,
        "blocker_reasons": blockers,
        "metrics": metrics,
        "gate_config": {
            "max_projection_ratio": max_projection_ratio,
            "max_projection_coeff": max_projection_coeff,
        },
    }


def _lane_provenance_from_candidate(candidate: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    composition = candidate.get("target_specific_composition") if isinstance(candidate.get("target_specific_composition"), dict) else {}
    provenance: dict[str, dict[str, Any]] = {}
    for route in composition.get("routes", []) if isinstance(composition.get("routes"), list) else []:
        if not isinstance(route, dict):
            continue
        lane = str(route.get("lane") or route.get("role") or "")
        role = str(route.get("role") or "")
        route_id = str(route.get("route_id") or "")
        for copied in route.get("copied_outputs", []) if isinstance(route.get("copied_outputs"), list) else []:
            if not isinstance(copied, dict):
                continue
            stem = str(copied.get("stem") or "")
            if not stem:
                continue
            provenance[stem] = {
                "lane": lane,
                "role": role,
                "route_id": route_id,
                "source_stem": str(copied.get("source_stem") or stem),
                "source_path": str(copied.get("source_path") or ""),
                "output_path": str(copied.get("output_path") or ""),
            }
    for stem in [*STEMS, "minus"]:
        provenance.setdefault(
            stem,
            {
                "lane": "candidate",
                "role": "candidate",
                "route_id": str(candidate.get("candidate_id") or ""),
                "source_stem": stem,
                "source_path": str((candidate.get("stem_paths") or {}).get(stem) or ""),
                "output_path": str((candidate.get("stem_paths") or {}).get(stem) or ""),
            },
        )
    return provenance


def build_listen_pack(
    *,
    output_dir: Path,
    mix_path: Path,
    windows: list[Mapping[str, Any]],
    accepted_window_candidates: Mapping[str, str],
) -> dict[str, Any]:
    expected_window_ids = [str(window["window_id"]) for window in windows]
    missing_windows = [window_id for window_id in expected_window_ids if not accepted_window_candidates.get(window_id)]
    if missing_windows:
        raise ValueError(f"listen pack requires accepted candidates for every window: {missing_windows}")

    listen_dir = output_dir / "listen"
    audio_dir = listen_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    source_paths: dict[str, list[Path]] = {stem: [] for stem in [*STEMS, "minus"]}
    lane_provenance: dict[str, dict[str, Any]] = {}
    window_lane_provenance: dict[str, dict[str, dict[str, Any]]] = {}
    mix_window_paths: list[Path] = []
    for window in windows:
        window_id = str(window["window_id"])
        candidate = _find_window_candidate(window, str(accepted_window_candidates[window_id]))
        stem_paths = candidate.get("stem_paths") if isinstance(candidate.get("stem_paths"), dict) else {}
        candidate_lane_provenance = _lane_provenance_from_candidate(candidate)
        window_lane_provenance[window_id] = {}
        for stem in source_paths:
            stem_path = stem_paths.get(stem)
            if not stem_path:
                raise ValueError(f"listen pack missing stem path: {window_id}:{stem}")
            source_paths[stem].append(Path(str(stem_path)))
            provenance = dict(candidate_lane_provenance.get(stem) or {})
            provenance["window_id"] = window_id
            provenance["candidate_id"] = str(candidate.get("candidate_id") or "")
            window_lane_provenance[window_id][stem] = provenance
            lane_provenance.setdefault(stem, {key: value for key, value in provenance.items() if key != "window_id"})
        mix_window_paths.append(Path(str(window["mix_path"])))

    audio_paths: dict[str, str] = {}
    for stem, paths in source_paths.items():
        audio, sample_rate = _concat_audio(paths)
        output_path = _write_audio(audio_dir / f"{stem}.flac", audio, sample_rate)
        audio_paths[stem] = str(output_path)
    mix_audio, mix_rate = _concat_audio(mix_window_paths)
    mix_output = _write_audio(audio_dir / "mix.flac", mix_audio, mix_rate)
    audio_paths["mix"] = str(mix_output)

    playlist_path = listen_dir / "playlist.m3u"
    playlist_path.write_text(
        "\n".join(f"audio/{name}.flac" for name in ["mix", "vocals", "minus", "drums", "bass", "guitar", "piano", "other"]) + "\n",
        encoding="utf-8",
    )
    pack = {
        "schema": "fully_automatic_stem_listen_pack.v1",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "source_mix_path": str(mix_path),
        "window_count": len(windows),
        "accepted_window_candidates": dict(accepted_window_candidates),
        "lane_provenance": lane_provenance,
        "window_lane_provenance": window_lane_provenance,
        "audio_paths": audio_paths,
        "playlist_path": str(playlist_path),
    }
    pack_path = listen_dir / "pack.json"
    _write_json(pack_path, pack)
    return {
        "path": str(pack_path),
        "audio_dir": str(audio_dir),
        "playlist_path": str(playlist_path),
        "audio_paths": audio_paths,
        "lane_provenance": lane_provenance,
        "window_lane_provenance": window_lane_provenance,
    }


def _score_candidate_contamination(
    *,
    candidate: Mapping[str, Any],
    candidate_dir: Path,
    stem_paths: Mapping[str, str],
) -> dict[str, Any]:
    references = candidate.get("foreign_references") if isinstance(candidate.get("foreign_references"), dict) else {}
    score_dir = candidate_dir / "score"
    blocker_reasons: list[str] = []
    leak_results: list[dict[str, Any]] = []
    attempted = 0
    checks, explicit_checks = _candidate_contamination_checks(candidate)

    for check in checks:
        tested_stem = check["tested_stem"]
        foreign_source = check["foreign_source"]
        if tested_stem not in FORBIDDEN_SOURCES or foreign_source not in FORBIDDEN_SOURCES[tested_stem]:
            blocker_reasons.append(f"invalid_contamination_check:{tested_stem}:{foreign_source}")
            continue
        stem_path = stem_paths.get(tested_stem)
        if not stem_path:
            blocker_reasons.append(f"missing_contamination_stem:{tested_stem}")
            continue
        source_refs = references.get(tested_stem) if isinstance(references.get(tested_stem), dict) else {}
        reference_path = source_refs.get(foreign_source) if isinstance(source_refs, dict) else None
        if not reference_path and not explicit_checks:
            blocker_reasons.append(f"missing_contamination_reference:{tested_stem}:{foreign_source}")
            continue

        attempted += 1
        result = score_stem_contamination(
            tested_stem_path=Path(stem_path),
            tested_stem=tested_stem,
            foreign_source=foreign_source,
            output_dir=score_dir,
            foreign_reference_path=Path(str(reference_path)) if reference_path else None,
            model_manifest_path=Path(str(check["model_manifest"])) if check.get("model_manifest") else None,
        )
        leak_results.append(result)
        if result.get("verdict") == "fail":
            blocker_reasons.append(f"foreign_source_recovered:{tested_stem}:{foreign_source}")
        elif result.get("verdict") == "blocked_unknown":
            reason = f"contamination_detector_unknown:{tested_stem}:{foreign_source}"
            blocker_reasons.append(reason)
            if result.get("reason"):
                blocker_reasons.append(f"{reason}:{result['reason']}")

    if any(reason.startswith("foreign_source_recovered:") for reason in blocker_reasons):
        verdict = "blocked_leak"
    elif blocker_reasons:
        verdict = "blocked_unknown"
    else:
        verdict = "auto_accept" if attempted > 0 else "blocked_unknown"

    return {
        "verdict": verdict,
        "attempted_checks": attempted,
        "blocker_reasons": blocker_reasons,
        "leak_results": leak_results,
        "check_mode": "explicit_only"
        if str(candidate.get("contamination_check_mode") or "").strip() == "explicit_only"
        else "full_forbidden_matrix",
        "candidate_check_hints_used": explicit_checks,
        "required_check_count": len(checks),
    }


def _score_candidate_cross_stem_forensic(
    *,
    candidate: Mapping[str, Any],
    candidate_dir: Path,
    stem_paths: Mapping[str, str],
) -> dict[str, Any]:
    config = _cross_stem_forensic_gate_config(candidate)
    if config.get("enabled") is False:
        return {
            "schema": "cross_stem_forensic_matrix.v1",
            "verdict": "blocked_unknown",
            "blocker_reasons": ["v2_cross_stem_forensic_gate_disabled"],
            "pair_results": [],
        }
    return score_cross_stem_forensic_matrix(
        stem_paths=stem_paths,
        output_dir=candidate_dir / "score" / "forensic",
        config=config,
    )


def _cross_stem_forensic_gate_config(owner: Mapping[str, Any]) -> dict[str, Any]:
    config = owner.get("cross_stem_forensic_gate") if isinstance(owner.get("cross_stem_forensic_gate"), dict) else {}
    return dict(config)


def _candidate_verdict_from_cross_stem_forensic(score: Mapping[str, Any]) -> str:
    if score.get("verdict") == "auto_accept":
        return "auto_accept"
    if score.get("verdict") == "fail":
        return "blocked_leak"
    return "blocked_unknown"


def _route_existing_audio_path_for_stem(route: Mapping[str, Any], stem: str, keys: Sequence[str]) -> Path | None:
    for copied in route.get("copied_outputs", []) if isinstance(route.get("copied_outputs"), list) else []:
        if not isinstance(copied, dict):
            continue
        if str(copied.get("stem") or "") != stem:
            continue
        for key in keys:
            audio_path = Path(str(copied.get(key) or ""))
            if audio_path.is_file():
                return audio_path
    return None


def _route_source_for_stem(route: Mapping[str, Any], stem: str) -> Path | None:
    return _route_existing_audio_path_for_stem(route, stem, ("source_path",))


def _route_pair_scoring_path_for_stem(route: Mapping[str, Any], stem: str) -> Path | None:
    return _route_existing_audio_path_for_stem(route, stem, ("output_path", "source_path"))


def _route_has_source_stems(route: Mapping[str, Any], stems: Sequence[str]) -> bool:
    return all(_route_source_for_stem(route, stem) is not None for stem in stems)


def _route_source_identity(route: Mapping[str, Any], stems: Sequence[str]) -> str:
    route_id = str(route.get("route_id") or route.get("route_alias") or route.get("stems_dir") or "")
    source_parts: list[str] = []
    for stem in stems:
        source_path = _route_source_for_stem(route, stem)
        source_parts.append(str(source_path.resolve()) if source_path is not None else "")
    return "\x1f".join([route_id, *source_parts])


def _model_split_route_priority(route: Mapping[str, Any]) -> tuple[int, str]:
    lane = str(route.get("lane") or "")
    route_id = str(route.get("route_id") or route.get("route_alias") or route.get("stems_dir") or "")
    is_teacher_mask_checkpoint = lane == "teacher_mask_checkpoint_lane" or route_id.startswith("tm")
    return (0 if is_teacher_mask_checkpoint else 1, route_id)


def _model_split_routes_by_role(
    window_candidates: Sequence[Mapping[str, Any]],
    role: str,
    required_stems: Sequence[str],
) -> list[Mapping[str, Any]]:
    routes: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for candidate in window_candidates:
        composition = candidate.get("target_specific_composition")
        if not isinstance(composition, dict):
            continue
        for route in composition.get("routes", []) if isinstance(composition.get("routes"), list) else []:
            if not isinstance(route, dict) or str(route.get("role") or "") != role:
                continue
            if route.get("blocker_reasons"):
                continue
            if not _route_has_source_stems(route, required_stems):
                continue
            key = _route_source_identity(route, required_stems)
            if key in seen:
                continue
            seen.add(key)
            routes.append(route)
    routes.sort(key=_model_split_route_priority)
    return routes


def _model_split_support_options(
    *,
    bass_support_routes: Sequence[Mapping[str, Any]],
    single_stem_support_routes: Mapping[str, Sequence[Mapping[str, Any]]],
    non_vocal_support_routes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Mapping[str, Any]]]:
    support_options: list[dict[str, Mapping[str, Any]]] = [{}]
    support_options.extend({"bass": route} for route in bass_support_routes)
    for stem in ("drums", "guitar", "piano", "other"):
        support_options.extend({stem: route} for route in single_stem_support_routes.get(stem, ()))
    support_options.extend({stem: route for stem in ("drums", "bass", "other")} for route in non_vocal_support_routes)

    combined_support: dict[str, Mapping[str, Any]] = {}
    if bass_support_routes:
        combined_support["bass"] = bass_support_routes[0]
    for stem in ("drums", "guitar", "piano", "other"):
        routes = single_stem_support_routes.get(stem, ())
        if routes:
            combined_support[stem] = routes[0]
    if len(combined_support) >= 2:
        support_options.append(combined_support)
    return support_options


def _route_id_for_pairing(route: Mapping[str, Any]) -> str:
    return str(route.get("route_id") or route.get("route_alias") or route.get("stems_dir") or "")


def _score_vocal_minus_route_pair(
    *,
    mix_path: Path,
    vocal_route: Mapping[str, Any],
    minus_route: Mapping[str, Any],
) -> float:
    vocal_path = _route_pair_scoring_path_for_stem(vocal_route, "vocals")
    minus_path = _route_pair_scoring_path_for_stem(minus_route, "minus")
    if vocal_path is None or minus_path is None:
        return float("inf")
    score = validate_mix_conservation(
        mix_path=mix_path,
        stem_paths={"vocals": str(vocal_path), "minus": str(minus_path)},
        required_stems=("vocals", "minus"),
        relative_tolerance=1.0,
    )
    if score.get("blockers"):
        return float("inf")
    return float(score.get("residual_ratio") or float("inf"))


def _rank_model_split_vocal_minus_pairs(
    *,
    mix_path: Path,
    vocal_routes: Sequence[Mapping[str, Any]],
    minus_routes: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any], float]]:
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any], float, int, int]] = []
    for vocal_index, vocal_route in enumerate(vocal_routes):
        for minus_index, minus_route in enumerate(minus_routes):
            residual_ratio = _score_vocal_minus_route_pair(
                mix_path=mix_path,
                vocal_route=vocal_route,
                minus_route=minus_route,
            )
            pairs.append((vocal_route, minus_route, residual_ratio, vocal_index, minus_index))
    return [
        (vocal_route, minus_route, residual_ratio)
        for vocal_route, minus_route, residual_ratio, _, _ in sorted(
            pairs,
            key=lambda item: (
                _model_split_route_priority(item[0])[0],
                _model_split_route_priority(item[1])[0],
                item[2],
                _route_id_for_pairing(item[0]),
                _route_id_for_pairing(item[1]),
                item[3],
                item[4],
            ),
        )
    ]


def _prioritized_model_split_support_repair_pairs(
    *,
    support_options: Sequence[Mapping[str, Mapping[str, Any]]],
    repair_options: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Mapping[str, Any]], Mapping[str, Any]]]:
    if not support_options:
        support_options = ({},)
    if not repair_options:
        repair_options = ({"repair_id": "none", "type": "none"},)

    base_support = support_options[0]
    support_tail = list(support_options[1:])
    first_repair = repair_options[0]
    pairs: list[tuple[Mapping[str, Mapping[str, Any]], Mapping[str, Any]]] = []
    seen: set[tuple[int, str, str]] = set()

    def add_pair(support_routes: Mapping[str, Mapping[str, Any]], repair_option: Mapping[str, Any]) -> None:
        key = (
            id(support_routes),
            str(repair_option.get("repair_id") or ""),
            str(repair_option.get("type") or ""),
        )
        if key in seen:
            return
        seen.add(key)
        pairs.append((support_routes, repair_option))

    add_pair(base_support, first_repair)

    priority_repair_ids = ("bs400", "pr16", "bs400_pr16", "vp100")
    learned_repairs: list[Mapping[str, Any]] = []
    priority_repairs: list[Mapping[str, Any]] = []
    remaining_repairs: list[Mapping[str, Any]] = []
    for repair_option in repair_options[1:]:
        repair_id = str(repair_option.get("repair_id") or "")
        repair_type = str(repair_option.get("type") or "")
        if "stem_region_repair" in repair_type.split("+"):
            learned_repairs.append(repair_option)
        elif repair_id in priority_repair_ids:
            priority_repairs.append(repair_option)
        else:
            remaining_repairs.append(repair_option)
    priority_repairs.sort(
        key=lambda repair_option: priority_repair_ids.index(str(repair_option.get("repair_id") or ""))
    )

    if learned_repairs and support_tail:
        add_pair(support_tail[0], first_repair)
    else:
        for support_routes in support_tail:
            add_pair(support_routes, first_repair)

    for repair_option in learned_repairs:
        for support_routes in support_tail:
            add_pair(support_routes, repair_option)
        add_pair(base_support, repair_option)

    if learned_repairs:
        for support_routes in support_tail:
            add_pair(support_routes, first_repair)

    for repair_option in priority_repairs:
        repair_id = str(repair_option.get("repair_id") or "")
        if repair_id in {"bs400", "bs400_pr16"}:
            bass_only_supports = [
                support_routes
                for support_routes in support_tail
                if "bass" in support_routes
                and not any(str(route.get("role") or "") == "non_vocal_support" for route in support_routes.values())
            ]
            for support_routes in bass_only_supports:
                add_pair(support_routes, repair_option)
            if not bass_only_supports:
                add_pair(base_support, repair_option)
            continue
        for support_routes in support_tail:
            add_pair(support_routes, repair_option)
        add_pair(base_support, repair_option)

    for repair_option in remaining_repairs:
        add_pair(base_support, repair_option)
    for support_routes in support_tail:
        for repair_option in remaining_repairs:
            add_pair(support_routes, repair_option)
    return pairs


def _copy_route_stem_to_candidate(
    *,
    route: Mapping[str, Any],
    stem: str,
    final_stems_dir: Path,
    window: Mapping[str, Any],
    copied_outputs: list[dict[str, str]],
) -> None:
    source_path = _route_source_for_stem(route, stem)
    if source_path is None:
        raise FileNotFoundError(f"missing composite source stem: {stem}")
    output_path = final_stems_dir / f"{stem}.flac"
    _copy_audio_to_match_shape(
        source_path,
        output_path,
        target_sample_rate=int(window["sample_rate"]) if window.get("sample_rate") is not None else None,
        target_sample_count=int(window["sample_count"]) if window.get("sample_count") is not None else None,
        target_channel_count=int(window["channel_count"]) if window.get("channel_count") is not None else None,
    )
    copied_outputs.append(
        {
            "stem": stem,
            "source_stem": stem,
            "source_path": str(source_path),
            "output_path": str(output_path),
        }
    )


def _composite_route_manifest(
    *,
    source_route: Mapping[str, Any],
    role: str,
    lane: str,
    copied_outputs: list[dict[str, str]],
) -> dict[str, Any]:
    route_id = str(source_route.get("route_id") or role)
    route_alias = str(source_route.get("route_alias") or _safe_artifact_segment(route_id, fallback=role))
    return {
        "role": role,
        "lane": lane,
        "route_id": route_id,
        "route_alias": route_alias,
        "source_candidate_id": str(source_route.get("source_candidate_id") or ""),
        "output_dir": str(source_route.get("output_dir") or ""),
        "stems_dir": str(source_route.get("stems_dir") or ""),
        "required_outputs": list(TARGET_SPECIFIC_ROLE_OUTPUTS.get(role, ())),
        "output_map": dict(source_route.get("output_map") or {}),
        "copied_outputs": copied_outputs,
        "missing_outputs": [],
        "blocker_reasons": [],
        "composite_source": True,
    }


def _finalize_composite_candidate(
    *,
    candidate: Mapping[str, Any],
    candidate_dir: Path,
    stem_paths: Mapping[str, str],
    target_specific_composition: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    damage_score: Mapping[str, Any] | None = None,
    voice_presence_score: Mapping[str, Any] | None = None,
    contamination_score: Mapping[str, Any] | None = None,
    cross_stem_forensic_score: Mapping[str, Any] | None = None,
    vocal_projection_debleed: Mapping[str, Any] | None = None,
    delivery_pair_gain_repair: Mapping[str, Any] | None = None,
    delivery_pair_residual_repair: Mapping[str, Any] | None = None,
    bass_stereo_width_repair: Mapping[str, Any] | None = None,
    stem_region_repair: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verdict = "auto_accept" if reconstruction.get("ok") else "blocked_bad_reconstruction"
    for gate_score in (damage_score, voice_presence_score, contamination_score):
        if gate_score and gate_score.get("verdict") != "auto_accept":
            verdict = str(gate_score.get("verdict") or "blocked_unknown")
            break
    else:
        if cross_stem_forensic_score is not None:
            verdict = _candidate_verdict_from_cross_stem_forensic(cross_stem_forensic_score)
    if not reconstruction.get("ok"):
        verdict = "blocked_bad_reconstruction"
    next_action = choose_next_action(
        {
            "verdict": verdict,
            "reason": ",".join(
                list((contamination_score or {}).get("blocker_reasons") or [])
                + list((cross_stem_forensic_score or {}).get("blocker_reasons") or [])
                + list((voice_presence_score or {}).get("blocker_reasons") or [])
                + list((damage_score or {}).get("blocker_reasons") or [])
                + list((reconstruction.get("delivery_pair") or {}).get("blockers") or [])
            ),
        }
    )
    manifest = build_candidate_manifest(str(candidate.get("candidate_id") or "c101"), candidate_dir)
    manifest.update(
        {
            "window_id": str(candidate.get("window_id") or ""),
            "execution_status": "completed",
            "candidate_untrusted": True,
            "model_split_composite": True,
            "bass_support_applied": bool(target_specific_composition.get("bass_support_applied")),
            "stem_support_applied": bool(target_specific_composition.get("stem_support_applied")),
            "non_vocal_support_applied": bool(target_specific_composition.get("non_vocal_support_applied")),
            "model_split_repair_id": str(candidate.get("model_split_repair_id") or "none"),
            "model_split_repair_type": str(candidate.get("model_split_repair_type") or "none"),
            "model_split_pair_residual_ratio": candidate.get("model_split_pair_residual_ratio"),
            "cross_stem_forensic_gate": _cross_stem_forensic_gate_config(candidate),
            "profile": MODEL_SPLIT_V2_PROFILE,
            "minus_source": "minus_lane",
            "minus_rebuilt_from_non_vocal_stems": False,
            "stems_dir": str(candidate_dir / "stems"),
            "score_path": str(candidate_dir / "score.json"),
            "required_stems": list(STEMS),
            "stem_paths": dict(stem_paths),
            "missing_stems": [],
            "reconstruction": reconstruction,
            "vocal_projection_debleed": vocal_projection_debleed,
            "delivery_pair_gain_repair": delivery_pair_gain_repair,
            "delivery_pair_residual_repair": delivery_pair_residual_repair,
            "bass_stereo_width_repair": bass_stereo_width_repair,
            "stem_region_repair": stem_region_repair,
            "target_specific_composition": target_specific_composition,
            "damage_score": damage_score,
            "voice_presence_score": voice_presence_score,
            "contamination_score": contamination_score,
            "cross_stem_forensic_score": cross_stem_forensic_score,
            "next_action": next_action,
            "verdict": verdict,
            "delivery_allowed": verdict == "auto_accept",
            "presentation_status": "accepted_for_listen_pack" if verdict == "auto_accept" else "blocked_from_listen_pack",
        }
    )
    score_payload = {
        "verdict": verdict,
        "reconstruction": reconstruction,
        "vocal_projection_debleed": vocal_projection_debleed,
        "delivery_pair_gain_repair": delivery_pair_gain_repair,
        "delivery_pair_residual_repair": delivery_pair_residual_repair,
        "bass_stereo_width_repair": bass_stereo_width_repair,
        "stem_region_repair": stem_region_repair,
        "model_split_pair_residual_ratio": candidate.get("model_split_pair_residual_ratio"),
        "damage_score": damage_score,
        "voice_presence_score": voice_presence_score,
        "contamination_score": contamination_score,
        "cross_stem_forensic_score": cross_stem_forensic_score,
        "next_action": next_action,
    }
    _write_json(candidate_dir / "score.json", score_payload)
    _write_json(candidate_dir / "manifest.json", manifest)
    return manifest


def build_model_split_v2_composite_candidates(
    *,
    window: Mapping[str, Any],
    window_dir: Path,
    window_candidates: Sequence[Mapping[str, Any]],
    candidate_template: Mapping[str, Any],
    max_composites: int = 324,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    stem_routes = _model_split_routes_by_role(window_candidates, "six_stems", NON_VOCAL_STEMS)
    vocal_routes = _model_split_routes_by_role(window_candidates, "vocals", ("vocals",))
    minus_routes = _model_split_routes_by_role(window_candidates, "minus", ("minus",))
    vocal_minus_pairs = _rank_model_split_vocal_minus_pairs(
        mix_path=Path(str(window.get("mix_path") or "")),
        vocal_routes=vocal_routes,
        minus_routes=minus_routes,
    )
    bass_support_routes = _model_split_routes_by_role(window_candidates, "bass_support", ("bass",))
    single_stem_support_routes = {
        "drums": _model_split_routes_by_role(window_candidates, "drums_support", ("drums",)),
        "guitar": _model_split_routes_by_role(window_candidates, "guitar_support", ("guitar",)),
        "piano": _model_split_routes_by_role(window_candidates, "piano_support", ("piano",)),
        "other": _model_split_routes_by_role(window_candidates, "other_support", ("other",)),
    }
    non_vocal_support_routes = _model_split_routes_by_role(window_candidates, "non_vocal_support", ("drums", "bass", "other"))
    support_options = _model_split_support_options(
        bass_support_routes=bass_support_routes,
        single_stem_support_routes=single_stem_support_routes,
        non_vocal_support_routes=non_vocal_support_routes,
    )
    repair_options_raw = candidate_template.get("model_split_composite_repair_options")
    repair_options: list[dict[str, Any]] = []
    if isinstance(repair_options_raw, list) and repair_options_raw:
        for raw_option in repair_options_raw:
            if not isinstance(raw_option, Mapping):
                continue
            repair_type = str(raw_option.get("type") or "none").strip() or "none"
            repair_id = _safe_artifact_segment(str(raw_option.get("repair_id") or repair_type), fallback="repair")
            if repair_type == "none":
                repair_options.append({"repair_id": repair_id, "type": "none"})
                continue
            repair_parts = repair_type.split("+")
            allowed_repair_parts = {
                "vocal_projection_debleed",
                "bass_stereo_width",
                "delivery_pair_gain",
                "delivery_pair_residual",
                "stem_region_repair",
            }
            if repair_parts and all(part in allowed_repair_parts for part in repair_parts):
                repair_option: dict[str, Any] = {"repair_id": repair_id, "type": repair_type}
                if len(repair_parts) == 1:
                    repair_config = dict(raw_option)
                    for key in ("repair_id", "type", "config"):
                        repair_config.pop(key, None)
                    if repair_type == "vocal_projection_debleed":
                        repair_config.setdefault("target_stems", list(NON_VOCAL_STEMS))
                    sequence_configs = raw_option.get("stem_region_repairs")
                    if repair_type == "stem_region_repair" and isinstance(sequence_configs, list):
                        repair_option["stem_region_repairs"] = [
                            dict(item) for item in sequence_configs if isinstance(item, Mapping)
                        ]
                    else:
                        repair_option[repair_type] = repair_config
                        repair_option["config"] = repair_config
                else:
                    for repair_part in repair_parts:
                        nested_config = raw_option.get(repair_part)
                        repair_config = dict(nested_config) if isinstance(nested_config, Mapping) else {}
                        if repair_part == "vocal_projection_debleed":
                            repair_config.setdefault("target_stems", list(NON_VOCAL_STEMS))
                        repair_option[repair_part] = repair_config
                    sequence_configs = raw_option.get("stem_region_repairs")
                    if isinstance(sequence_configs, list):
                        repair_option["stem_region_repairs"] = [
                            dict(item) for item in sequence_configs if isinstance(item, Mapping)
                        ]
                repair_options.append(repair_option)
                continue
            if repair_type == "bass_stereo_width":
                repair_config = dict(raw_option)
                for key in ("repair_id", "type", "config"):
                    repair_config.pop(key, None)
                repair_options.append(
                    {
                        "repair_id": repair_id,
                        "type": repair_type,
                        "config": repair_config,
                    }
                )
                continue
            if repair_type == "vocal_projection_debleed+bass_stereo_width":
                vocal_config = raw_option.get("vocal_projection_debleed")
                bass_config = raw_option.get("bass_stereo_width")
                repair_options.append(
                    {
                        "repair_id": repair_id,
                        "type": repair_type,
                        "vocal_projection_debleed": dict(vocal_config) if isinstance(vocal_config, Mapping) else {},
                        "bass_stereo_width": dict(bass_config) if isinstance(bass_config, Mapping) else {},
                    }
                )
                continue
            if repair_type != "vocal_projection_debleed":
                continue
            nested_config = raw_option.get("config")
            repair_config = dict(nested_config) if isinstance(nested_config, Mapping) else dict(raw_option)
            for key in ("repair_id", "type", "config"):
                repair_config.pop(key, None)
            repair_config.setdefault("target_stems", list(NON_VOCAL_STEMS))
            repair_options.append(
                {
                    "repair_id": repair_id,
                    "type": repair_type,
                    "config": repair_config,
                }
            )
    if not repair_options:
        repair_options = [{"repair_id": "none", "type": "none"}]
    composite_candidates: list[dict[str, Any]] = []

    def _emit_progress(event: dict[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback(event)

    def _append_composite(manifest: dict[str, Any]) -> None:
        composite_candidates.append(manifest)
        _emit_progress(
            {
                "event": "composite_candidate_end",
                "window_id": str(window.get("window_id") or ""),
                "candidate_id": manifest.get("candidate_id"),
                "verdict": manifest.get("verdict"),
                "repair_id": manifest.get("model_split_repair_id"),
                "route_family": manifest.get("route_family"),
            }
        )

    composite_index = 0
    for stem_route in stem_routes:
        for vocal_route, minus_route, pair_residual_ratio in vocal_minus_pairs:
            for support_routes, repair_option in _prioritized_model_split_support_repair_pairs(
                support_options=support_options,
                repair_options=repair_options,
            ):
                    composite_index += 1
                    if composite_index > max_composites:
                        _emit_progress(
                            {
                                "event": "composite_candidate_limit_reached",
                                "window_id": str(window.get("window_id") or ""),
                                "max_composites": max_composites,
                            }
                        )
                        return composite_candidates
                    repair_type = str(repair_option.get("type") or "none")
                    repair_id = _safe_artifact_segment(str(repair_option.get("repair_id") or repair_type), fallback="repair")
                    candidate_id = f"c{100 + composite_index:03d}"
                    _emit_progress(
                        {
                            "event": "composite_candidate_start",
                            "window_id": str(window.get("window_id") or ""),
                            "candidate_id": candidate_id,
                            "repair_id": repair_id,
                            "repair_type": repair_type,
                            "composite_index": composite_index,
                            "max_composites": max_composites,
                            "bass_support_applied": "bass" in support_routes,
                            "stem_support_applied": bool(support_routes),
                            "non_vocal_support_applied": any(
                                str(route.get("role") or "") == "non_vocal_support" for route in support_routes.values()
                            ),
                            "pair_residual_ratio": round(float(pair_residual_ratio), 8)
                            if math.isfinite(pair_residual_ratio)
                            else None,
                        }
                    )
                    candidate_dir = window_dir / candidate_id
                    stems_dir = candidate_dir / "stems"
                    stems_dir.mkdir(parents=True, exist_ok=True)
                    candidate = dict(candidate_template)
                    candidate.update(
                        {
                            "candidate_id": candidate_id,
                            "window_id": str(window.get("window_id") or ""),
                            "profile": MODEL_SPLIT_V2_PROFILE,
                            "minus_source": "minus_lane",
                            "minus_rebuilt_from_non_vocal_stems": False,
                            "route_family": "model_split_v2_composite",
                            "bass_support_applied": "bass" in support_routes,
                            "stem_support_applied": bool(support_routes),
                            "non_vocal_support_applied": any(
                                str(route.get("role") or "") == "non_vocal_support" for route in support_routes.values()
                            ),
                            "model_split_repair_id": repair_id,
                            "model_split_repair_type": repair_type,
                            "model_split_pair_residual_ratio": pair_residual_ratio if math.isfinite(pair_residual_ratio) else None,
                        }
                    )
                    repair_parts = set(repair_type.split("+"))
                    if "vocal_projection_debleed" in repair_parts:
                        candidate["vocal_projection_debleed"] = dict(repair_option.get("vocal_projection_debleed") or {})
                    if "delivery_pair_gain" in repair_parts:
                        candidate["delivery_pair_gain_repair"] = dict(repair_option.get("delivery_pair_gain") or {})
                    if "delivery_pair_residual" in repair_parts:
                        candidate["delivery_pair_residual_repair"] = dict(repair_option.get("delivery_pair_residual") or {})
                    if "bass_stereo_width" in repair_parts:
                        candidate["bass_stereo_width_repair"] = dict(repair_option.get("bass_stereo_width") or {})
                    if "stem_region_repair" in repair_parts:
                        candidate["stem_region_repair"] = dict(repair_option.get("stem_region_repair") or {})
                    stem_copies: list[dict[str, str]] = []
                    vocal_copies: list[dict[str, str]] = []
                    minus_copies: list[dict[str, str]] = []
                    support_copies_by_key: dict[str, list[dict[str, str]]] = {}
                    support_routes_by_key: dict[str, Mapping[str, Any]] = {}
                    try:
                        for stem in NON_VOCAL_STEMS:
                            _copy_route_stem_to_candidate(
                                route=stem_route,
                                stem=stem,
                                final_stems_dir=stems_dir,
                                window=window,
                                copied_outputs=stem_copies,
                            )
                        for stem, support_route in support_routes.items():
                            support_key = str(
                                support_route.get("route_id")
                                or support_route.get("route_alias")
                                or support_route.get("stems_dir")
                                or id(support_route)
                            )
                            support_routes_by_key[support_key] = support_route
                            support_copies = support_copies_by_key.setdefault(support_key, [])
                            _copy_route_stem_to_candidate(
                                route=support_route,
                                stem=stem,
                                final_stems_dir=stems_dir,
                                window=window,
                                copied_outputs=support_copies,
                            )
                        _copy_route_stem_to_candidate(
                            route=vocal_route,
                            stem="vocals",
                            final_stems_dir=stems_dir,
                            window=window,
                            copied_outputs=vocal_copies,
                        )
                        _copy_route_stem_to_candidate(
                            route=minus_route,
                            stem="minus",
                            final_stems_dir=stems_dir,
                            window=window,
                            copied_outputs=minus_copies,
                        )
                    except Exception as exc:
                        manifest = build_candidate_manifest(candidate_id, candidate_dir)
                        next_action = choose_next_action({"verdict": "blocked_unknown", "reason": "composite_copy_failed"})
                        manifest.update(
                            {
                                "window_id": str(window.get("window_id") or ""),
                                "execution_status": "failed_runtime_error",
                                "model_split_composite": True,
                                "model_split_repair_id": repair_id,
                                "model_split_repair_type": repair_type,
                                "profile": MODEL_SPLIT_V2_PROFILE,
                                "verdict": "blocked_unknown",
                                "error": {"type": type(exc).__name__, "message": str(exc)},
                                "next_action": next_action,
                                "delivery_allowed": False,
                                "presentation_status": "blocked_from_listen_pack",
                            }
                        )
                        _write_json(candidate_dir / "manifest.json", manifest)
                        _write_json(candidate_dir / "score.json", {"verdict": "blocked_unknown", "next_action": next_action})
                        _append_composite(manifest)
                        continue

                    routes = [
                        _composite_route_manifest(
                            source_route=stem_route,
                            role="six_stems",
                            lane="stem_lane",
                            copied_outputs=stem_copies,
                        ),
                        _composite_route_manifest(
                            source_route=vocal_route,
                            role="vocals",
                            lane="vocal_lane",
                            copied_outputs=vocal_copies,
                        ),
                        _composite_route_manifest(
                            source_route=minus_route,
                            role="minus",
                            lane="minus_lane",
                            copied_outputs=minus_copies,
                        ),
                    ]
                    for support_key, support_copies in support_copies_by_key.items():
                        support_route = support_routes_by_key[support_key]
                        routes.append(
                            _composite_route_manifest(
                                source_route=support_route,
                                role=str(support_route.get("role") or "stem_support"),
                                lane=str(support_route.get("lane") or "stem_support_lane"),
                                copied_outputs=support_copies,
                            )
                        )
                    target_specific_composition = {
                        "applied": True,
                        "composite_lane_selection": True,
                        "bass_support_applied": "bass" in support_routes,
                        "stem_support_applied": bool(support_routes),
                        "non_vocal_support_applied": any(
                            str(route.get("role") or "") == "non_vocal_support" for route in support_routes.values()
                        ),
                        "repair": {"repair_id": repair_id, "type": repair_type},
                        "blocker_reasons": [],
                        "routes": routes,
                    }
                    stem_paths = _collect_stem_paths(stems_dir)
                    missing = missing_required_stems(stem_paths) + ([] if stem_paths.get("minus") else ["minus"])
                    vocal_projection_debleed: dict[str, Any] | None = None
                    delivery_pair_gain_repair: dict[str, Any] | None = None
                    delivery_pair_residual_repair: dict[str, Any] | None = None
                    bass_stereo_width_repair: dict[str, Any] | None = None
                    stem_region_repair: dict[str, Any] | None = None
                    if missing:
                        reconstruction = {
                            "ok": False,
                            "mode": "model_split_v2_delivery_pair",
                            "gated_surface": "vocals_plus_minus",
                            "delivery_pair": {"blockers": [f"missing_reconstruction_stem:{stem}" for stem in missing]},
                            "six_stem_diagnostic": {"ok": False, "blockers": ["missing_composite_stems"]},
                        }
                        manifest = _finalize_composite_candidate(
                            candidate=candidate,
                            candidate_dir=candidate_dir,
                            stem_paths=stem_paths,
                            target_specific_composition=target_specific_composition,
                            reconstruction=reconstruction,
                        )
                        manifest["missing_stems"] = missing
                        _write_json(candidate_dir / "manifest.json", manifest)
                        _append_composite(manifest)
                        continue

                    if "stem_region_repair" in repair_parts and "vocal_projection_debleed" in repair_parts:
                        stem_region_sequence = repair_option.get("stem_region_repairs")
                        if isinstance(stem_region_sequence, list) and stem_region_sequence:
                            stem_region_repair = apply_learned_stem_region_repairs(
                                mix_path=Path(str(window["mix_path"])),
                                stem_paths=stem_paths,
                                configs=[dict(item) for item in stem_region_sequence if isinstance(item, Mapping)],
                            )
                        else:
                            stem_region_config = repair_option.get("stem_region_repair") or repair_option.get("config")
                            stem_region_repair = apply_learned_stem_region_repair(
                                mix_path=Path(str(window["mix_path"])),
                                stem_paths=stem_paths,
                                config=dict(stem_region_config or {}),
                            )
                        repair_manifest = dict(target_specific_composition.get("repair") or {})
                        repair_manifest.update(
                            {
                                "repair_id": repair_id,
                                "type": repair_type,
                                "stem_region_repair": stem_region_repair,
                            }
                        )
                        target_specific_composition["repair"] = repair_manifest
                        if stem_region_repair.get("blockers"):
                            reconstruction = {
                                "ok": False,
                                "mode": "model_split_v2_delivery_pair",
                                "gated_surface": "vocals_plus_minus",
                                "delivery_pair": {
                                    "blockers": [
                                        f"composite_repair_blocker:{blocker}"
                                        for blocker in stem_region_repair.get("blockers", [])
                                    ]
                                },
                                "six_stem_diagnostic": {"ok": False, "blockers": ["composite_repair_failed"]},
                            }
                            _append_composite(
                                _finalize_composite_candidate(
                                    candidate=candidate,
                                    candidate_dir=candidate_dir,
                                    stem_paths=stem_paths,
                                    target_specific_composition=target_specific_composition,
                                    reconstruction=reconstruction,
                                    stem_region_repair=stem_region_repair,
                                )
                            )
                            continue

                    if "vocal_projection_debleed" in repair_parts:
                        projection_config = repair_option.get("vocal_projection_debleed") or repair_option.get("config")
                        vocal_projection_debleed = apply_vocal_projection_debleed(
                            stem_paths=stem_paths,
                            config=dict(projection_config or {}),
                        )
                        target_specific_composition["repair"] = {
                            "repair_id": repair_id,
                            "type": repair_type,
                            "vocal_projection_debleed": vocal_projection_debleed,
                        }
                        if vocal_projection_debleed.get("blockers"):
                            reconstruction = {
                                "ok": False,
                                "mode": "model_split_v2_delivery_pair",
                                "gated_surface": "vocals_plus_minus",
                                "delivery_pair": {
                                    "blockers": [
                                        f"composite_repair_blocker:{blocker}"
                                        for blocker in vocal_projection_debleed.get("blockers", [])
                                    ]
                                },
                                "six_stem_diagnostic": {"ok": False, "blockers": ["composite_repair_failed"]},
                            }
                            _append_composite(
                                _finalize_composite_candidate(
                                    candidate=candidate,
                                    candidate_dir=candidate_dir,
                                    stem_paths=stem_paths,
                                    target_specific_composition=target_specific_composition,
                                    reconstruction=reconstruction,
                                    vocal_projection_debleed=vocal_projection_debleed,
                                )
                            )
                            continue
                    if "delivery_pair_gain" in repair_parts:
                        pair_config = repair_option.get("delivery_pair_gain") or repair_option.get("config")
                        delivery_pair_gain_repair = apply_delivery_pair_gain_repair(
                            mix_path=Path(str(window["mix_path"])),
                            stem_paths=stem_paths,
                            config=dict(pair_config or {}),
                        )
                        repair_manifest = dict(target_specific_composition.get("repair") or {})
                        repair_manifest.update(
                            {
                                "repair_id": repair_id,
                                "type": repair_type,
                                "delivery_pair_gain_repair": delivery_pair_gain_repair,
                            }
                        )
                        if vocal_projection_debleed is not None:
                            repair_manifest["vocal_projection_debleed"] = vocal_projection_debleed
                        target_specific_composition["repair"] = repair_manifest
                        if delivery_pair_gain_repair.get("blockers"):
                            reconstruction = {
                                "ok": False,
                                "mode": "model_split_v2_delivery_pair",
                                "gated_surface": "vocals_plus_minus",
                                "delivery_pair": {
                                    "blockers": [
                                        f"composite_repair_blocker:{blocker}"
                                        for blocker in delivery_pair_gain_repair.get("blockers", [])
                                    ]
                                },
                                "six_stem_diagnostic": {"ok": False, "blockers": ["composite_repair_failed"]},
                            }
                            _append_composite(
                                _finalize_composite_candidate(
                                    candidate=candidate,
                                    candidate_dir=candidate_dir,
                                    stem_paths=stem_paths,
                                    target_specific_composition=target_specific_composition,
                                    reconstruction=reconstruction,
                                    vocal_projection_debleed=vocal_projection_debleed,
                                    delivery_pair_gain_repair=delivery_pair_gain_repair,
                                )
                            )
                            continue
                    if "delivery_pair_residual" in repair_parts:
                        residual_config = repair_option.get("delivery_pair_residual") or repair_option.get("config")
                        residual_config_dict = dict(residual_config or {})
                        residual_blocker = _delivery_pair_residual_blocker_after_stem_region_repair(
                            stem_region_repair,
                            residual_config_dict,
                        )
                        if residual_blocker:
                            delivery_pair_residual_repair = {
                                "applied": False,
                                "repair_owner": "runner_delivery_pair_residual_repair_v1",
                                "reason": "delivery_pair_residual_blocked_after_stem_region_repair",
                                "target_stem": str(residual_config_dict.get("target_stem") or "minus"),
                                "blockers": [residual_blocker],
                            }
                        else:
                            delivery_pair_residual_repair = apply_delivery_pair_residual_repair(
                                mix_path=Path(str(window["mix_path"])),
                                stem_paths=stem_paths,
                                config=residual_config_dict,
                            )
                        repair_manifest = dict(target_specific_composition.get("repair") or {})
                        repair_manifest.update(
                            {
                                "repair_id": repair_id,
                                "type": repair_type,
                                "delivery_pair_residual_repair": delivery_pair_residual_repair,
                            }
                        )
                        if vocal_projection_debleed is not None:
                            repair_manifest["vocal_projection_debleed"] = vocal_projection_debleed
                        if delivery_pair_gain_repair is not None:
                            repair_manifest["delivery_pair_gain_repair"] = delivery_pair_gain_repair
                        if delivery_pair_residual_repair is not None:
                            repair_manifest["delivery_pair_residual_repair"] = delivery_pair_residual_repair
                        target_specific_composition["repair"] = repair_manifest
                        if delivery_pair_residual_repair.get("blockers"):
                            reconstruction = {
                                "ok": False,
                                "mode": "model_split_v2_delivery_pair",
                                "gated_surface": "vocals_plus_minus",
                                "delivery_pair": {
                                    "blockers": [
                                        f"composite_repair_blocker:{blocker}"
                                        for blocker in delivery_pair_residual_repair.get("blockers", [])
                                    ]
                                },
                                "six_stem_diagnostic": {"ok": False, "blockers": ["composite_repair_failed"]},
                            }
                            _append_composite(
                                _finalize_composite_candidate(
                                    candidate=candidate,
                                    candidate_dir=candidate_dir,
                                    stem_paths=stem_paths,
                                    target_specific_composition=target_specific_composition,
                                    reconstruction=reconstruction,
                                    vocal_projection_debleed=vocal_projection_debleed,
                                    delivery_pair_gain_repair=delivery_pair_gain_repair,
                                    delivery_pair_residual_repair=delivery_pair_residual_repair,
                                )
                            )
                            continue
                    if "bass_stereo_width" in repair_parts:
                        bass_config = repair_option.get("bass_stereo_width") or repair_option.get("config")
                        bass_stereo_width_repair = apply_bass_stereo_width_repair(
                            stem_paths=stem_paths,
                            config=dict(bass_config or {}),
                        )
                        repair_manifest = dict(target_specific_composition.get("repair") or {})
                        repair_manifest.update(
                            {
                                "repair_id": repair_id,
                                "type": repair_type,
                                "bass_stereo_width_repair": bass_stereo_width_repair,
                            }
                        )
                        if vocal_projection_debleed is not None:
                            repair_manifest["vocal_projection_debleed"] = vocal_projection_debleed
                        if delivery_pair_gain_repair is not None:
                            repair_manifest["delivery_pair_gain_repair"] = delivery_pair_gain_repair
                        target_specific_composition["repair"] = repair_manifest
                        if bass_stereo_width_repair.get("blockers"):
                            reconstruction = {
                                "ok": False,
                                "mode": "model_split_v2_delivery_pair",
                                "gated_surface": "vocals_plus_minus",
                                "delivery_pair": {
                                    "blockers": [
                                        f"composite_repair_blocker:{blocker}"
                                        for blocker in bass_stereo_width_repair.get("blockers", [])
                                    ]
                                },
                                "six_stem_diagnostic": {"ok": False, "blockers": ["composite_repair_failed"]},
                            }
                            _append_composite(
                                _finalize_composite_candidate(
                                    candidate=candidate,
                                    candidate_dir=candidate_dir,
                                    stem_paths=stem_paths,
                                    target_specific_composition=target_specific_composition,
                                    reconstruction=reconstruction,
                                    vocal_projection_debleed=vocal_projection_debleed,
                                    delivery_pair_gain_repair=delivery_pair_gain_repair,
                                    delivery_pair_residual_repair=delivery_pair_residual_repair,
                                    bass_stereo_width_repair=bass_stereo_width_repair,
                                )
                            )
                            continue
                    if "stem_region_repair" in repair_parts and stem_region_repair is None:
                        stem_region_sequence = repair_option.get("stem_region_repairs")
                        if isinstance(stem_region_sequence, list) and stem_region_sequence:
                            stem_region_repair = apply_learned_stem_region_repairs(
                                mix_path=Path(str(window["mix_path"])),
                                stem_paths=stem_paths,
                                configs=[dict(item) for item in stem_region_sequence if isinstance(item, Mapping)],
                            )
                        else:
                            stem_region_config = repair_option.get("stem_region_repair") or repair_option.get("config")
                            stem_region_repair = apply_learned_stem_region_repair(
                                mix_path=Path(str(window["mix_path"])),
                                stem_paths=stem_paths,
                                config=dict(stem_region_config or {}),
                            )
                        repair_manifest = dict(target_specific_composition.get("repair") or {})
                        repair_manifest.update(
                            {
                                "repair_id": repair_id,
                                "type": repair_type,
                                "stem_region_repair": stem_region_repair,
                            }
                        )
                        if vocal_projection_debleed is not None:
                            repair_manifest["vocal_projection_debleed"] = vocal_projection_debleed
                        if delivery_pair_gain_repair is not None:
                            repair_manifest["delivery_pair_gain_repair"] = delivery_pair_gain_repair
                        if delivery_pair_residual_repair is not None:
                            repair_manifest["delivery_pair_residual_repair"] = delivery_pair_residual_repair
                        if bass_stereo_width_repair is not None:
                            repair_manifest["bass_stereo_width_repair"] = bass_stereo_width_repair
                        target_specific_composition["repair"] = repair_manifest
                        if stem_region_repair.get("blockers"):
                            reconstruction = {
                                "ok": False,
                                "mode": "model_split_v2_delivery_pair",
                                "gated_surface": "vocals_plus_minus",
                                "delivery_pair": {
                                    "blockers": [
                                        f"composite_repair_blocker:{blocker}"
                                        for blocker in stem_region_repair.get("blockers", [])
                                    ]
                                },
                                "six_stem_diagnostic": {"ok": False, "blockers": ["composite_repair_failed"]},
                            }
                            _append_composite(
                                _finalize_composite_candidate(
                                    candidate=candidate,
                                    candidate_dir=candidate_dir,
                                    stem_paths=stem_paths,
                                    target_specific_composition=target_specific_composition,
                                    reconstruction=reconstruction,
                                    vocal_projection_debleed=vocal_projection_debleed,
                                    delivery_pair_gain_repair=delivery_pair_gain_repair,
                                    delivery_pair_residual_repair=delivery_pair_residual_repair,
                                    bass_stereo_width_repair=bass_stereo_width_repair,
                                    stem_region_repair=stem_region_repair,
                                )
                            )
                            continue

                    reconstruction_config = candidate.get("delivery_pair_reconstruction")
                    reconstruction = validate_model_split_v2_conservation(
                        mix_path=Path(str(window["mix_path"])),
                        stem_paths=stem_paths,
                        target_specific_composition=target_specific_composition,
                        config=reconstruction_config if isinstance(reconstruction_config, Mapping) else None,
                    )
                    if not reconstruction["ok"]:
                        _append_composite(
                            _finalize_composite_candidate(
                                candidate=candidate,
                                candidate_dir=candidate_dir,
                                stem_paths=stem_paths,
                                target_specific_composition=target_specific_composition,
                                reconstruction=reconstruction,
                                vocal_projection_debleed=vocal_projection_debleed,
                                delivery_pair_gain_repair=delivery_pair_gain_repair,
                                delivery_pair_residual_repair=delivery_pair_residual_repair,
                                bass_stereo_width_repair=bass_stereo_width_repair,
                                stem_region_repair=stem_region_repair,
                            )
                        )
                        continue
                    damage_score = score_model_split_v2_damage_gates(
                        candidate=candidate,
                        mix_path=Path(str(window["mix_path"])),
                        stem_paths=stem_paths,
                    )
                    if damage_score["verdict"] != "auto_accept":
                        _append_composite(
                            _finalize_composite_candidate(
                                candidate=candidate,
                                candidate_dir=candidate_dir,
                                stem_paths=stem_paths,
                                target_specific_composition=target_specific_composition,
                                reconstruction=reconstruction,
                                damage_score=damage_score,
                                vocal_projection_debleed=vocal_projection_debleed,
                                delivery_pair_gain_repair=delivery_pair_gain_repair,
                                delivery_pair_residual_repair=delivery_pair_residual_repair,
                                bass_stereo_width_repair=bass_stereo_width_repair,
                                stem_region_repair=stem_region_repair,
                            )
                        )
                        continue
                    voice_presence_score = score_model_split_v2_voice_presence_gate(candidate=candidate, stem_paths=stem_paths)
                    if voice_presence_score["verdict"] != "auto_accept":
                        _append_composite(
                            _finalize_composite_candidate(
                                candidate=candidate,
                                candidate_dir=candidate_dir,
                                stem_paths=stem_paths,
                                target_specific_composition=target_specific_composition,
                                reconstruction=reconstruction,
                                damage_score=damage_score,
                                voice_presence_score=voice_presence_score,
                                vocal_projection_debleed=vocal_projection_debleed,
                                delivery_pair_gain_repair=delivery_pair_gain_repair,
                                delivery_pair_residual_repair=delivery_pair_residual_repair,
                                bass_stereo_width_repair=bass_stereo_width_repair,
                                stem_region_repair=stem_region_repair,
                            )
                        )
                        continue
                    contamination_score = _score_candidate_contamination(
                        candidate=candidate,
                        candidate_dir=candidate_dir,
                        stem_paths=stem_paths,
                    )
                    cross_stem_forensic_score = (
                        _score_candidate_cross_stem_forensic(
                            candidate=candidate,
                            candidate_dir=candidate_dir,
                            stem_paths=stem_paths,
                        )
                        if contamination_score["verdict"] == "auto_accept"
                        else None
                    )
                    _append_composite(
                        _finalize_composite_candidate(
                            candidate=candidate,
                            candidate_dir=candidate_dir,
                            stem_paths=stem_paths,
                            target_specific_composition=target_specific_composition,
                            reconstruction=reconstruction,
                            damage_score=damage_score,
                            voice_presence_score=voice_presence_score,
                            contamination_score=contamination_score,
                            cross_stem_forensic_score=cross_stem_forensic_score,
                            vocal_projection_debleed=vocal_projection_debleed,
                            delivery_pair_gain_repair=delivery_pair_gain_repair,
                            delivery_pair_residual_repair=delivery_pair_residual_repair,
                            bass_stereo_width_repair=bass_stereo_width_repair,
                            stem_region_repair=stem_region_repair,
                        )
                    )
                    if composite_candidates[-1].get("verdict") == "auto_accept":
                        return composite_candidates
    return composite_candidates


def execute_recipe_candidate(
    *,
    candidate: Mapping[str, Any],
    window: Mapping[str, Any],
    window_dir: Path,
    command_runner: CommandRunner,
    timeout_sec: int,
    route_cache: dict[str, dict[str, Any]] | None = None,
    separator_backend_config: SeparatorBackendConfig | None = None,
    separator_client: PersistentSeparatorClient | None = None,
    route_budget_profile: str = ROUTE_BUDGET_PROFILE_FULL,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "c001")
    candidate_dir = window_dir / candidate_id
    stems_dir = candidate_dir / "stems"
    manifest = build_candidate_manifest(candidate_id, candidate_dir)
    manifest.update(
        {
            "window_id": window["window_id"],
            "execution_status": "pending",
            "stems_dir": str(stems_dir),
            "score_path": str(candidate_dir / "score.json"),
            "profile": _profile_name(candidate),
            "minus_source": _candidate_minus_source(candidate),
            "minus_rebuilt_from_non_vocal_stems": bool(candidate.get("minus_rebuilt_from_non_vocal_stems", False)),
            "required_stems": list(_candidate_required_stems(candidate)),
        }
    )
    command_template = candidate.get("command_template")
    target_specific_commands = _target_specific_items(candidate)
    if not target_specific_commands and (not isinstance(command_template, list) or not command_template):
        next_action = choose_next_action({"verdict": "blocked_unknown", "reason": "missing_command_template"})
        manifest.update(
            {
                "execution_status": "skipped_missing_command_template",
                "verdict": "blocked_unknown",
                "next_action": next_action,
            }
        )
        _write_json(candidate_dir / "manifest.json", manifest)
        _write_json(candidate_dir / "score.json", {"verdict": "blocked_unknown", "reason": "missing_command_template", "next_action": next_action})
        return manifest

    stems_dir.mkdir(parents=True, exist_ok=True)
    target_specific_composition: dict[str, Any] = {"applied": False, "routes": [], "blocker_reasons": []}
    delivery_pair_fail_fast: dict[str, Any] | None = None
    try:
        if target_specific_commands and _can_model_split_v2_delivery_pair_fail_fast(candidate):
            delivery_items, remaining_items = _split_model_split_v2_delivery_pair_items(candidate)
            target_specific_composition = _run_target_specific_commands(
                candidate=_with_target_specific_items(candidate, delivery_items),
                window=window,
                candidate_dir=candidate_dir,
                stems_dir=stems_dir,
                command_runner=command_runner,
                timeout_sec=timeout_sec,
                route_cache=route_cache,
                separator_backend_config=separator_backend_config,
                separator_client=separator_client,
                route_budget_profile=route_budget_profile,
            )
            stem_paths = _collect_stem_paths(stems_dir)
            precheck = validate_model_split_v2_conservation(
                mix_path=Path(str(window["mix_path"])),
                stem_paths=stem_paths,
                target_specific_composition=target_specific_composition,
                config=candidate.get("delivery_pair_reconstruction")
                if isinstance(candidate.get("delivery_pair_reconstruction"), Mapping)
                else None,
            )
            delivery_pair_fail_fast = {
                "enabled": True,
                "checked_after_roles": sorted(MODEL_SPLIT_V2_DELIVERY_PAIR_ROLES),
                "skipped_roles": [str(item.get("role") or "") for item in remaining_items],
                "reconstruction": precheck,
            }
            if not precheck.get("ok"):
                next_action = choose_next_action({"verdict": "blocked_bad_reconstruction", "reason": "delivery_pair_fail_fast"})
                manifest.update(
                    {
                        "execution_status": "completed",
                        "candidate_untrusted": True,
                        "verdict": "blocked_bad_reconstruction",
                        "stem_paths": stem_paths,
                        "missing_stems": [],
                        "reconstruction": precheck,
                        "delivery_pair_fail_fast": delivery_pair_fail_fast,
                        "target_specific_composition": target_specific_composition,
                        "next_action": next_action,
                        "delivery_allowed": False,
                        "presentation_status": "blocked_from_listen_pack",
                        "minus_source": _candidate_minus_source(candidate),
                        "minus_rebuilt_from_non_vocal_stems": False,
                    }
                )
                _write_json(
                    candidate_dir / "score.json",
                    {
                        "verdict": "blocked_bad_reconstruction",
                        "reconstruction": precheck,
                        "delivery_pair_fail_fast": delivery_pair_fail_fast,
                        "target_specific_composition": target_specific_composition,
                        "next_action": next_action,
                    },
                )
                _write_json(candidate_dir / "manifest.json", manifest)
                return manifest
            if remaining_items:
                remaining_composition = _run_target_specific_commands(
                    candidate=_with_target_specific_items(candidate, remaining_items),
                    window=window,
                    candidate_dir=candidate_dir,
                    stems_dir=stems_dir,
                    command_runner=command_runner,
                    timeout_sec=timeout_sec,
                    route_cache=route_cache,
                    separator_backend_config=separator_backend_config,
                    separator_client=separator_client,
                    route_budget_profile=route_budget_profile,
                )
                target_specific_composition = _merge_target_specific_compositions(
                    target_specific_composition,
                    remaining_composition,
                )
        elif target_specific_commands:
            target_specific_composition = _run_target_specific_commands(
                candidate=candidate,
                window=window,
                candidate_dir=candidate_dir,
                stems_dir=stems_dir,
                command_runner=command_runner,
                timeout_sec=timeout_sec,
                route_cache=route_cache,
                separator_backend_config=separator_backend_config,
                separator_client=separator_client,
                route_budget_profile=route_budget_profile,
            )
        else:
            replacements = {
                "window_id": str(window["window_id"]),
                "input_audio": str(window["mix_path"]),
                "window_mix": str(window["mix_path"]),
                "output_dir": str(candidate_dir),
                "stems_dir": str(stems_dir),
                "candidate_id": candidate_id,
            }
            command = render_recipe_command(command_template, replacements)
            manifest["command"] = command
            setattr(command_runner, "stems_dir", stems_dir)
            command_runner(command, timeout_sec)
    except subprocess.TimeoutExpired as exc:
        next_action = choose_next_action({"verdict": "blocked_unknown", "reason": "failed_timeout"})
        manifest.update(
            {
                "execution_status": "failed_timeout",
                "verdict": "blocked_unknown",
                "next_action": next_action,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _write_json(candidate_dir / "manifest.json", manifest)
        _write_json(candidate_dir / "score.json", {"verdict": "blocked_unknown", "reason": "failed_timeout", "next_action": next_action})
        return manifest
    except PersistentSeparatorUnavailable as exc:
        next_action = choose_next_action({"verdict": "blocked_unknown", "reason": "separator_server_unavailable"})
        manifest.update(
            {
                "execution_status": "failed_separator_backend_unavailable",
                "verdict": "blocked_unknown",
                "root_cause": "infra_gap",
                "next_action": next_action,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "delivery_allowed": False,
                "presentation_status": "blocked_from_listen_pack",
            }
        )
        _write_json(candidate_dir / "manifest.json", manifest)
        _write_json(
            candidate_dir / "score.json",
            {
                "verdict": "blocked_unknown",
                "reason": "separator_server_unavailable",
                "root_cause": "infra_gap",
                "next_action": next_action,
            },
        )
        return manifest
    except PersistentSeparatorRejected as exc:
        next_action = choose_next_action({"verdict": "blocked_unknown", "reason": "separator_server_rejected"})
        manifest.update(
            {
                "execution_status": "failed_separator_backend_rejected",
                "verdict": "blocked_unknown",
                "root_cause": "infra_gap",
                "next_action": next_action,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "delivery_allowed": False,
                "presentation_status": "blocked_from_listen_pack",
            }
        )
        _write_json(candidate_dir / "manifest.json", manifest)
        _write_json(
            candidate_dir / "score.json",
            {
                "verdict": "blocked_unknown",
                "reason": "separator_server_rejected",
                "root_cause": "infra_gap",
                "next_action": next_action,
            },
        )
        return manifest
    except Exception as exc:
        next_action = choose_next_action({"verdict": "blocked_unknown", "reason": "failed_runtime_error"})
        manifest.update(
            {
                "execution_status": "failed_runtime_error",
                "verdict": "blocked_unknown",
                "next_action": next_action,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _write_json(candidate_dir / "manifest.json", manifest)
        _write_json(candidate_dir / "score.json", {"verdict": "blocked_unknown", "reason": "failed_runtime_error", "next_action": next_action})
        return manifest

    stem_paths = _collect_stem_paths(stems_dir)
    if target_specific_composition["blocker_reasons"]:
        next_action = choose_next_action(
            {"verdict": "blocked_missing_stem", "reason": ",".join(target_specific_composition["blocker_reasons"])}
        )
        manifest.update(
            {
                "execution_status": "completed",
                "verdict": "blocked_missing_stem",
                "missing_stems": _missing_candidate_required_stems(candidate, stem_paths),
                "stem_paths": stem_paths,
                "target_specific_composition": target_specific_composition,
                "next_action": next_action,
                "delivery_allowed": False,
                "presentation_status": "blocked_from_listen_pack",
            }
        )
        _write_json(candidate_dir / "manifest.json", manifest)
        _write_json(
            candidate_dir / "score.json",
            {
                "verdict": "blocked_missing_stem",
                "target_specific_composition": target_specific_composition,
                "next_action": next_action,
            },
        )
        return manifest
    candidate_required_stems = _candidate_required_stems(candidate)
    missing = _missing_candidate_required_stems(candidate, stem_paths)
    if missing:
        next_action = choose_next_action({"verdict": "blocked_missing_stem", "reason": "missing_required_stem"})
        manifest.update(
            {
                "execution_status": "completed",
                "verdict": "blocked_missing_stem",
                "missing_stems": missing,
                "stem_paths": stem_paths,
                "target_specific_composition": target_specific_composition,
                "next_action": next_action,
                "required_stems": list(candidate_required_stems),
                "delivery_allowed": False,
                "presentation_status": "blocked_from_listen_pack",
            }
        )
        _write_json(candidate_dir / "manifest.json", manifest)
        _write_json(candidate_dir / "score.json", {"verdict": "blocked_missing_stem", "missing_stems": missing, "next_action": next_action})
        return manifest

    debleed_config = candidate.get("minus_vocal_debleed")
    minus_vocal_debleed: dict[str, Any] | None = None
    if isinstance(debleed_config, dict):
        minus_vocal_debleed = apply_minus_vocal_debleed(stem_paths=stem_paths, config=debleed_config)

    projection_debleed_config = candidate.get("vocal_projection_debleed")
    vocal_projection_debleed: dict[str, Any] | None = None
    if isinstance(projection_debleed_config, dict):
        vocal_projection_debleed = apply_vocal_projection_debleed(
            stem_paths=stem_paths,
            config=projection_debleed_config,
        )

    repair_stem = str(candidate.get("conservation_repair_stem") or "").strip()
    reconstruction_repair: dict[str, Any] | None = None
    if repair_stem:
        max_repair_ratio_raw = candidate.get("max_conservation_repair_ratio")
        max_repair_ratio = float(max_repair_ratio_raw) if max_repair_ratio_raw is not None else None
        reconstruction_repair = apply_mix_conservation_repair(
            mix_path=Path(str(window["mix_path"])),
            stem_paths=stem_paths,
            target_stem=repair_stem,
            max_residual_ratio=max_repair_ratio,
        )

    stem_merges: dict[str, Any] | None = None
    merge_specs = candidate.get("stem_merges")
    if isinstance(merge_specs, list) and merge_specs:
        stem_merges = apply_stem_merges(stem_paths=stem_paths, merge_specs=merge_specs)
        if stem_merges.get("blocker_reasons"):
            next_action = choose_next_action(
                {"verdict": "blocked_unknown", "reason": ",".join(stem_merges.get("blocker_reasons") or [])}
            )
            manifest.update(
                {
                    "execution_status": "completed",
                    "candidate_untrusted": True,
                    "verdict": "blocked_unknown",
                    "stem_paths": stem_paths,
                    "missing_stems": [],
                    "reconstruction_repair": reconstruction_repair,
                    "minus_vocal_debleed": minus_vocal_debleed,
                    "vocal_projection_debleed": vocal_projection_debleed,
                    "stem_merges": stem_merges,
                    "target_specific_composition": target_specific_composition,
                    "next_action": next_action,
                    "delivery_allowed": False,
                    "presentation_status": "blocked_from_listen_pack",
                }
            )
            _write_json(
                candidate_dir / "score.json",
                {
                    "verdict": "blocked_unknown",
                    "stem_merges": stem_merges,
                    "next_action": next_action,
                },
            )
            _write_json(candidate_dir / "manifest.json", manifest)
            return manifest

    post_debleed_config = candidate.get("post_minus_vocal_debleed")
    post_minus_vocal_debleed: dict[str, Any] | None = None
    if isinstance(post_debleed_config, dict):
        post_minus_vocal_debleed = apply_minus_vocal_debleed(stem_paths=stem_paths, config=post_debleed_config)
        if post_minus_vocal_debleed.get("blockers"):
            next_action = choose_next_action(
                {
                    "verdict": "blocked_unknown",
                    "reason": ",".join(post_minus_vocal_debleed.get("blockers") or []),
                }
            )
            manifest.update(
                {
                    "execution_status": "completed",
                    "candidate_untrusted": True,
                    "verdict": "blocked_unknown",
                    "stem_paths": stem_paths,
                    "missing_stems": [],
                    "reconstruction_repair": reconstruction_repair,
                    "minus_vocal_debleed": minus_vocal_debleed,
                    "vocal_projection_debleed": vocal_projection_debleed,
                    "stem_merges": stem_merges,
                    "post_minus_vocal_debleed": post_minus_vocal_debleed,
                    "target_specific_composition": target_specific_composition,
                    "next_action": next_action,
                    "delivery_allowed": False,
                    "presentation_status": "blocked_from_listen_pack",
                }
            )
            _write_json(
                candidate_dir / "score.json",
                {
                    "verdict": "blocked_unknown",
                    "post_minus_vocal_debleed": post_minus_vocal_debleed,
                    "next_action": next_action,
                },
            )
            _write_json(candidate_dir / "manifest.json", manifest)
            return manifest

    minus_rebuilt_from_non_vocal_stems = bool(candidate.get("minus_rebuilt_from_non_vocal_stems", False))
    if _is_model_split_v2(candidate) or str(candidate.get("minus_source") or "") == "minus_lane":
        minus_rebuilt_from_non_vocal_stems = False
    if minus_rebuilt_from_non_vocal_stems:
        next_action = choose_next_action(
            {"verdict": "blocked_missing_stem", "reason": "stem_derived_minus_is_not_allowed"}
        )
        manifest.update(
            {
                "execution_status": "completed",
                "candidate_untrusted": True,
                "verdict": "blocked_missing_stem",
                "missing_stems": ["minus"],
                "stem_paths": stem_paths,
                "reconstruction_repair": reconstruction_repair,
                "minus_vocal_debleed": minus_vocal_debleed,
                "vocal_projection_debleed": vocal_projection_debleed,
                "stem_merges": stem_merges,
                "post_minus_vocal_debleed": post_minus_vocal_debleed,
                "target_specific_composition": target_specific_composition,
                "next_action": next_action,
                "delivery_allowed": False,
                "presentation_status": "blocked_from_listen_pack",
                "minus_source": _candidate_minus_source(candidate),
                "minus_rebuilt_from_non_vocal_stems": False,
                "blocker_reason": "stem_derived_minus_is_not_allowed",
            }
        )
        _write_json(
            candidate_dir / "score.json",
            {
                "verdict": "blocked_missing_stem",
                "missing_stems": ["minus"],
                "blocker_reason": "stem_derived_minus_is_not_allowed",
                "next_action": next_action,
            },
        )
        _write_json(candidate_dir / "manifest.json", manifest)
        return manifest
    elif not stem_paths.get("minus"):
        damage_score = score_model_split_v2_damage_gates(
            candidate=candidate,
            mix_path=Path(str(window["mix_path"])),
            stem_paths=stem_paths,
            stem_merges=stem_merges,
            minus_vocal_debleed=minus_vocal_debleed,
            post_minus_vocal_debleed=post_minus_vocal_debleed,
        )
        if damage_score["verdict"] != "auto_accept":
            next_action = choose_next_action(
                {
                    "verdict": damage_score["verdict"],
                    "reason": ",".join(damage_score.get("blocker_reasons") or []),
                }
            )
            manifest.update(
                {
                    "execution_status": "completed",
                    "candidate_untrusted": True,
                    "verdict": damage_score["verdict"],
                    "missing_stems": ["minus"],
                    "stem_paths": stem_paths,
                    "reconstruction_repair": reconstruction_repair,
                    "minus_vocal_debleed": minus_vocal_debleed,
                    "vocal_projection_debleed": vocal_projection_debleed,
                    "stem_merges": stem_merges,
                    "post_minus_vocal_debleed": post_minus_vocal_debleed,
                    "target_specific_composition": target_specific_composition,
                    "damage_score": damage_score,
                    "next_action": next_action,
                    "delivery_allowed": False,
                    "presentation_status": "blocked_from_listen_pack",
                    "minus_source": _candidate_minus_source(candidate),
                    "minus_rebuilt_from_non_vocal_stems": False,
                }
            )
            _write_json(
                candidate_dir / "score.json",
                {"verdict": damage_score["verdict"], "damage_score": damage_score, "next_action": next_action},
            )
            _write_json(candidate_dir / "manifest.json", manifest)
            return manifest
        next_action = choose_next_action({"verdict": "blocked_missing_stem", "reason": "missing_minus_lane_output"})
        manifest.update(
            {
                "execution_status": "completed",
                "candidate_untrusted": True,
                "verdict": "blocked_missing_stem",
                "missing_stems": ["minus"],
                "stem_paths": stem_paths,
                "reconstruction_repair": reconstruction_repair,
                "minus_vocal_debleed": minus_vocal_debleed,
                "vocal_projection_debleed": vocal_projection_debleed,
                "stem_merges": stem_merges,
                "post_minus_vocal_debleed": post_minus_vocal_debleed,
                "delivery_pair_fail_fast": delivery_pair_fail_fast,
                "target_specific_composition": target_specific_composition,
                "next_action": next_action,
                "delivery_allowed": False,
                "presentation_status": "blocked_from_listen_pack",
                "minus_source": _candidate_minus_source(candidate),
                "minus_rebuilt_from_non_vocal_stems": False,
            }
        )
        _write_json(candidate_dir / "score.json", {"verdict": "blocked_missing_stem", "missing_stems": ["minus"], "next_action": next_action})
        _write_json(candidate_dir / "manifest.json", manifest)
        return manifest
    if _is_model_split_v2(candidate):
        reconstruction_config = candidate.get("delivery_pair_reconstruction")
        reconstruction = validate_model_split_v2_conservation(
            mix_path=Path(str(window["mix_path"])),
            stem_paths=stem_paths,
            target_specific_composition=target_specific_composition,
            config=reconstruction_config if isinstance(reconstruction_config, Mapping) else None,
        )
    else:
        reconstruction = validate_mix_conservation(
            mix_path=Path(str(window["mix_path"])),
            stem_paths=stem_paths,
            required_stems=candidate_required_stems,
        )
    if not reconstruction["ok"]:
        next_action = choose_next_action({"verdict": "blocked_bad_reconstruction", "reason": "mix_conservation_failed"})
        manifest.update(
            {
                "execution_status": "completed",
                "candidate_untrusted": True,
                "verdict": "blocked_bad_reconstruction",
                "stem_paths": stem_paths,
                "missing_stems": [],
                "reconstruction": reconstruction,
                "reconstruction_repair": reconstruction_repair,
                "minus_vocal_debleed": minus_vocal_debleed,
                "vocal_projection_debleed": vocal_projection_debleed,
                "stem_merges": stem_merges,
                "post_minus_vocal_debleed": post_minus_vocal_debleed,
                "target_specific_composition": target_specific_composition,
                "next_action": next_action,
                "delivery_allowed": False,
                "presentation_status": "blocked_from_listen_pack",
                "minus_source": _candidate_minus_source(candidate),
                "minus_rebuilt_from_non_vocal_stems": minus_rebuilt_from_non_vocal_stems,
            }
        )
        _write_json(
            candidate_dir / "score.json",
            {
                "verdict": "blocked_bad_reconstruction",
                "reconstruction": reconstruction,
                "reconstruction_repair": reconstruction_repair,
                "minus_vocal_debleed": minus_vocal_debleed,
                "vocal_projection_debleed": vocal_projection_debleed,
                "stem_merges": stem_merges,
                "post_minus_vocal_debleed": post_minus_vocal_debleed,
                "delivery_pair_fail_fast": delivery_pair_fail_fast,
                "target_specific_composition": target_specific_composition,
                "next_action": next_action,
            },
        )
        _write_json(candidate_dir / "manifest.json", manifest)
        return manifest

    damage_score = score_model_split_v2_damage_gates(
        candidate=candidate,
        mix_path=Path(str(window["mix_path"])),
        stem_paths=stem_paths,
        stem_merges=stem_merges,
        minus_vocal_debleed=minus_vocal_debleed,
        post_minus_vocal_debleed=post_minus_vocal_debleed,
    )
    if damage_score["verdict"] != "auto_accept":
        next_action = choose_next_action(
            {
                "verdict": damage_score["verdict"],
                "reason": ",".join(damage_score.get("blocker_reasons") or []),
            }
        )
        manifest.update(
            {
                "execution_status": "completed",
                "candidate_untrusted": True,
                "verdict": damage_score["verdict"],
                "stem_paths": stem_paths,
                "missing_stems": [],
                "reconstruction": reconstruction,
                "reconstruction_repair": reconstruction_repair,
                "minus_vocal_debleed": minus_vocal_debleed,
                "vocal_projection_debleed": vocal_projection_debleed,
                "stem_merges": stem_merges,
                "post_minus_vocal_debleed": post_minus_vocal_debleed,
                "target_specific_composition": target_specific_composition,
                "damage_score": damage_score,
                "next_action": next_action,
                "delivery_allowed": False,
                "presentation_status": "blocked_from_listen_pack",
                "minus_source": _candidate_minus_source(candidate),
                "minus_rebuilt_from_non_vocal_stems": minus_rebuilt_from_non_vocal_stems,
            }
        )
        _write_json(candidate_dir / "score.json", {"verdict": damage_score["verdict"], "damage_score": damage_score, "next_action": next_action})
        _write_json(candidate_dir / "manifest.json", manifest)
        return manifest

    voice_presence_score = score_model_split_v2_voice_presence_gate(candidate=candidate, stem_paths=stem_paths)
    if voice_presence_score["verdict"] != "auto_accept":
        next_action = choose_next_action(
            {
                "verdict": voice_presence_score["verdict"],
                "reason": ",".join(voice_presence_score.get("blocker_reasons") or []),
            }
        )
        manifest.update(
            {
                "execution_status": "completed",
                "candidate_untrusted": True,
                "verdict": voice_presence_score["verdict"],
                "stem_paths": stem_paths,
                "missing_stems": [],
                "reconstruction": reconstruction,
                "reconstruction_repair": reconstruction_repair,
                "minus_vocal_debleed": minus_vocal_debleed,
                "vocal_projection_debleed": vocal_projection_debleed,
                "stem_merges": stem_merges,
                "post_minus_vocal_debleed": post_minus_vocal_debleed,
                "target_specific_composition": target_specific_composition,
                "damage_score": damage_score,
                "voice_presence_score": voice_presence_score,
                "next_action": next_action,
                "delivery_allowed": False,
                "presentation_status": "blocked_from_listen_pack",
                "minus_source": _candidate_minus_source(candidate),
                "minus_rebuilt_from_non_vocal_stems": minus_rebuilt_from_non_vocal_stems,
            }
        )
        _write_json(
            candidate_dir / "score.json",
            {"verdict": voice_presence_score["verdict"], "voice_presence_score": voice_presence_score, "next_action": next_action},
        )
        _write_json(candidate_dir / "manifest.json", manifest)
        return manifest

    score = _score_candidate_contamination(candidate=candidate, candidate_dir=candidate_dir, stem_paths=stem_paths)
    cross_stem_forensic_score = (
        _score_candidate_cross_stem_forensic(candidate=candidate, candidate_dir=candidate_dir, stem_paths=stem_paths)
        if score["verdict"] == "auto_accept"
        else None
    )
    final_verdict = (
        _candidate_verdict_from_cross_stem_forensic(cross_stem_forensic_score)
        if cross_stem_forensic_score is not None and cross_stem_forensic_score["verdict"] != "auto_accept"
        else score["verdict"]
    )
    next_action = choose_next_action(
        {
            "verdict": final_verdict,
            "reason": ",".join(list(score.get("blocker_reasons") or []) + list((cross_stem_forensic_score or {}).get("blocker_reasons") or [])),
        }
    )
    manifest.update(
        {
            "execution_status": "completed",
            "candidate_untrusted": final_verdict != "auto_accept" or bool(candidate.get("diagnostic_only", False)),
            "verdict": final_verdict,
            "stem_paths": stem_paths,
            "missing_stems": [],
            "required_stems": list(candidate_required_stems),
            "reconstruction": reconstruction,
            "reconstruction_repair": reconstruction_repair,
            "minus_vocal_debleed": minus_vocal_debleed,
            "vocal_projection_debleed": vocal_projection_debleed,
            "stem_merges": stem_merges,
            "post_minus_vocal_debleed": post_minus_vocal_debleed,
            "target_specific_composition": target_specific_composition,
            "cross_stem_forensic_gate": _cross_stem_forensic_gate_config(candidate),
            "damage_score": damage_score,
            "voice_presence_score": voice_presence_score,
            "contamination_score": score,
            "cross_stem_forensic_score": cross_stem_forensic_score,
            "next_action": next_action,
            "delivery_allowed": final_verdict == "auto_accept" and not bool(candidate.get("diagnostic_only", False)),
            "presentation_status": "accepted_for_listen_pack"
            if final_verdict == "auto_accept" and not bool(candidate.get("diagnostic_only", False))
            else "blocked_from_listen_pack",
            "minus_source": _candidate_minus_source(candidate),
            "minus_rebuilt_from_non_vocal_stems": minus_rebuilt_from_non_vocal_stems,
        }
    )
    score_payload = {
        **score,
        "verdict": final_verdict,
        "contamination_score": score,
        "cross_stem_forensic_score": cross_stem_forensic_score,
        "next_action": next_action,
    }
    _write_json(candidate_dir / "score.json", score_payload)
    _write_json(candidate_dir / "manifest.json", manifest)
    return manifest


def choose_next_action(issue: Mapping[str, Any]) -> dict[str, Any]:
    verdict = str(issue.get("verdict") or "blocked_unknown")
    reason = str(issue.get("reason") or "")
    if verdict == "blocked_leak" and "vocal" in reason and "minus" in reason:
        action = "retry_stronger_vocal_removal"
    elif verdict == "blocked_leak" and ("guitar" in reason or "piano" in reason):
        action = "retry_alternate_six_stem_model"
    elif verdict == "blocked_damage" and "thin_vocal" in reason:
        action = "retry_less_aggressive_vocal_route"
    elif verdict == "blocked_damage" and "minus" in reason:
        action = "reject_subtraction_heavy_candidate"
    elif verdict == "blocked_unknown":
        action = "block"
    else:
        action = "retry_once_then_block"
    return {"action": action, "verdict": verdict if verdict in VERDICTS else "blocked_unknown", "reason": reason}


def select_accepted_candidate(
    candidates: list[Mapping[str, Any]],
    *,
    required_window_ids: list[str] | None = None,
) -> dict[str, Any]:
    def can_deliver(candidate: Mapping[str, Any]) -> bool:
        return candidate.get("verdict") == "auto_accept" and candidate.get("delivery_allowed", True) is not False

    if required_window_ids is not None:
        if not required_window_ids:
            return {
                "verdict": "blocked_no_clean_candidate",
                "accepted_candidate_id": "",
                "accepted_window_candidates": {},
                "blocked_window_ids": [],
                "delivery_allowed": False,
            }
        accepted_by_window: dict[str, str] = {}
        blocked_window_ids: list[str] = []
        for window_id in required_window_ids:
            accepted = next(
                (
                    candidate
                    for candidate in candidates
                    if str(candidate.get("window_id") or "") == window_id and can_deliver(candidate)
                ),
                None,
            )
            if accepted is None:
                blocked_window_ids.append(window_id)
                continue
            accepted_by_window[window_id] = str(accepted.get("candidate_id") or "")
        if blocked_window_ids:
            return {
                "verdict": "blocked_no_clean_candidate",
                "accepted_candidate_id": "",
                "accepted_window_candidates": accepted_by_window,
                "blocked_window_ids": blocked_window_ids,
                "delivery_allowed": False,
            }
        candidate_ids = sorted(set(accepted_by_window.values()))
        return {
            "verdict": "auto_accept",
            "accepted_candidate_id": candidate_ids[0] if len(candidate_ids) == 1 else "window_set",
            "accepted_window_candidates": accepted_by_window,
            "blocked_window_ids": [],
            "delivery_allowed": bool(required_window_ids),
        }

    for candidate in candidates:
        if can_deliver(candidate):
            return {
                "verdict": "auto_accept",
                "accepted_candidate_id": str(candidate.get("candidate_id") or ""),
                "accepted_window_candidates": {},
                "blocked_window_ids": [],
                "delivery_allowed": True,
            }
    return {
        "verdict": "blocked_no_clean_candidate",
        "accepted_candidate_id": "",
        "accepted_window_candidates": {},
        "blocked_window_ids": [],
        "delivery_allowed": False,
    }


def _write_run_manifest_with_training_queue(output_dir: Path, run_manifest: dict[str, Any]) -> Path:
    run_json_path = _write_json(output_dir / "run.json", run_manifest)
    if bool(run_manifest.get("delivery_allowed")):
        return run_json_path
    queue_manifest = build_root_cause_queue(
        run_json_path=run_json_path,
        output_dir=output_dir / "training" / "root_cause",
        min_repeats=2,
    )
    run_manifest["training_first_root_cause_queue_manifest"] = str(output_dir / "training" / "root_cause" / "manifest.json")
    run_manifest["training_first_root_cause_queue_jsonl"] = str(output_dir / "training" / "root_cause" / "queue.jsonl")
    run_manifest["training_first_root_cause_queue_rows"] = int(queue_manifest.get("queue_row_count") or 0)
    run_manifest["training_first_root_cause_counts"] = queue_manifest.get("root_cause_counts") or {}
    run_manifest["training_first_issue_counts"] = queue_manifest.get("issue_type_counts") or {}
    return _write_json(output_dir / "run.json", run_manifest)


def _effective_model_split_composite_limit(
    *,
    recipe: Mapping[str, Any],
    max_candidates: int,
    default_recipe: bool,
) -> int:
    configured = int(recipe.get("model_split_composite_max_candidates") or 72)
    if not default_recipe:
        return configured
    return max(0, min(configured, int(max_candidates)))


def run_fully_automatic_stem_search(
    *,
    mix_path: Path,
    output_dir: Path,
    window_sec: float = 6.0,
    max_candidates: int = 10,
    require_gpu: bool = True,
    no_artifact_rag: bool = True,
    recipe_path: Path | None = None,
    command_runner: CommandRunner | None = None,
    allow_default_recipe: bool = True,
    demucs_available: bool | None = None,
    contamination_model_manifest: Path | None = None,
    profile: str = "",
    stem_region_repair_checkpoints: Sequence[Mapping[str, Any]] | None = None,
    teacher_mask_checkpoints: Sequence[Mapping[str, Any]] | None = None,
    root_cause_queue_jsonl: Path | None = None,
    route_policy_checkpoint: Path | None = None,
    route_policy_mode: str = "shadow",
    separator_backend: str = MODE_SUBPROCESS,
    separator_server_url: str = "",
    separator_device: str = "",
    separator_request_timeout_sec: float = 0.0,
    min_gpu_free_mib: int = 0,
    wait_for_gpu_free_sec: float = 0.0,
    gpu_preflight_poll_sec: float = 5.0,
    route_budget_profile: str = ROUTE_BUDGET_PROFILE_FULL,
) -> dict[str, Any]:
    if no_artifact_rag:
        validate_no_artifact_rag_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    separator_backend_config = make_separator_backend_config(
        mode=separator_backend,
        server_url=separator_server_url,
        require_gpu=require_gpu,
        device=separator_device or ("cuda" if require_gpu else "cpu"),
        request_timeout_sec=separator_request_timeout_sec,
    )
    _append_jsonl(
        progress_path,
        {
            "event": "start",
            "elapsed_sec": 0.0,
            "mix_path": str(mix_path),
            "window_sec": window_sec,
            "max_candidates": max_candidates,
            "gpu_required": require_gpu,
            "artifact_rag_write_allowed": not no_artifact_rag,
            "profile": profile or "default",
            "root_cause_queue_jsonl": str(root_cause_queue_jsonl or ""),
            "route_policy_checkpoint": str(route_policy_checkpoint or ""),
            "route_policy_mode": str(route_policy_mode or "shadow"),
            "route_budget_profile": str(route_budget_profile or ROUTE_BUDGET_PROFILE_FULL),
            "separator_backend": separator_backend_config.manifest(),
            "teacher_mask_checkpoint_count": len(teacher_mask_checkpoints or []),
            "min_gpu_free_mib": int(max(0, min_gpu_free_mib)),
            "wait_for_gpu_free_sec": float(max(0.0, wait_for_gpu_free_sec)),
            "gpu_preflight_poll_sec": float(max(0.1, gpu_preflight_poll_sec)),
        },
    )

    started = time.monotonic()
    candidates: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    recipe: dict[str, Any] | None = None
    preloaded_root_cause_route_policy = _load_root_cause_route_policy(root_cause_queue_jsonl)
    cuda_preflight = _collect_separator_accelerator_preflight(
        require_gpu=require_gpu,
        min_free_mib=min_gpu_free_mib,
        separator_backend_config=separator_backend_config,
    )
    wait_deadline = time.monotonic() + float(max(0.0, wait_for_gpu_free_sec))
    poll_sec = float(max(0.1, gpu_preflight_poll_sec))
    while (
        require_gpu
        and cuda_preflight.get("status") == "blocked"
        and float(max(0.0, wait_for_gpu_free_sec)) > 0.0
        and time.monotonic() < wait_deadline
    ):
        _append_jsonl(
            progress_path,
            {
                "event": "gpu_preflight_wait",
                "elapsed_sec": round(time.monotonic() - started, 6),
                "poll_sec": poll_sec,
                "deadline_remaining_sec": round(max(0.0, wait_deadline - time.monotonic()), 6),
                **cuda_preflight,
            },
        )
        time.sleep(min(poll_sec, max(0.0, wait_deadline - time.monotonic())))
        cuda_preflight = _collect_separator_accelerator_preflight(
            require_gpu=require_gpu,
            min_free_mib=min_gpu_free_mib,
            separator_backend_config=separator_backend_config,
        )
    _append_jsonl(
        progress_path,
        {
            "event": "gpu_preflight",
            "elapsed_sec": round(time.monotonic() - started, 6),
            **cuda_preflight,
        },
    )
    if require_gpu and cuda_preflight.get("status") == "blocked":
        run_manifest = {
            "schema": "fully_automatic_stem_search_run.v1",
            "mix_path": str(mix_path),
            "output_dir": str(output_dir),
            "window_sec": window_sec,
            "max_candidates": max_candidates,
            "gpu_required": require_gpu,
            "artifact_rag_write_allowed": not no_artifact_rag,
            "profile": profile or "default",
            "root_cause_route_policy": preloaded_root_cause_route_policy,
            "learned_route_policy": {},
            "route_policy_checkpoint": str(route_policy_checkpoint or ""),
            "route_policy_mode": str(route_policy_mode or "shadow"),
            "route_budget_profile": str(route_budget_profile or ROUTE_BUDGET_PROFILE_FULL),
            "separator_backend": separator_backend_config.manifest(),
            "gpu_preflight": cuda_preflight,
            "required_stems": list(STEMS),
            "candidate_count": 0,
            "candidates": [],
            "accepted_window_candidates": {},
            "verdict": "blocked_unknown",
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "failed_checks": ["gpu_preflight_failed"],
            "root_cause": "infra_gap",
        }
        _write_json(output_dir / "run.json", run_manifest)
        _append_jsonl(
            progress_path,
            {
                "event": "end",
                "elapsed_sec": round(time.monotonic() - started, 6),
                "verdict": "blocked_unknown",
                "reason": cuda_preflight.get("reason") or "gpu_preflight_failed",
            },
        )
        return run_manifest
    learned_route_policy_model = _load_learned_route_policy_checkpoint(route_policy_checkpoint)
    if recipe_path is not None:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8-sig"))
    elif allow_default_recipe:
        model_manifest = contamination_model_manifest or _discover_default_contamination_model_manifest()
        if profile == MODEL_SPLIT_V2_PROFILE:
            recipe = build_model_split_v2_recipe(
                demucs_available=demucs_available,
                contamination_model_manifest=model_manifest,
                stem_region_repair_checkpoints=stem_region_repair_checkpoints,
                teacher_mask_checkpoints=teacher_mask_checkpoints,
            )
        elif separator_backend_config.enabled and separator_backend_config.device == "mps":
            recipe = build_persistent_mps_minus_pair_recipe(
                contamination_model_manifest=model_manifest,
            )
        else:
            recipe = build_default_from_scratch_recipe(
                demucs_available=demucs_available,
                contamination_model_manifest=model_manifest,
            )
    if recipe is not None and contamination_model_manifest is not None and not recipe.get("contamination_model_manifest"):
        recipe["contamination_model_manifest"] = str(contamination_model_manifest)
    if recipe is not None and root_cause_queue_jsonl is not None:
        recipe = _apply_root_cause_route_policy(recipe, root_cause_queue_jsonl=root_cause_queue_jsonl)
    if recipe is not None and learned_route_policy_model is not None:
        recipe = _apply_learned_route_policy_to_recipe(
            recipe,
            route_policy_model=learned_route_policy_model,
            route_policy_mode=route_policy_mode,
        )

    if recipe is not None and recipe.get("available", True) is False:
        run_manifest = {
            "schema": "fully_automatic_stem_search_run.v1",
            "mix_path": str(mix_path),
            "output_dir": str(output_dir),
            "window_sec": window_sec,
            "max_candidates": max_candidates,
            "gpu_required": require_gpu,
            "artifact_rag_write_allowed": not no_artifact_rag,
            "profile": profile or str((recipe or {}).get("profile") or "default"),
            "root_cause_route_policy": (recipe or {}).get("root_cause_route_policy") or {},
            "learned_route_policy": (recipe or {}).get("learned_route_policy") or {},
            "route_policy_checkpoint": str(route_policy_checkpoint or ""),
            "route_policy_mode": str(route_policy_mode or "shadow"),
            "route_budget_profile": str(route_budget_profile or ROUTE_BUDGET_PROFILE_FULL),
            "separator_backend": separator_backend_config.manifest(),
            "gpu_preflight": cuda_preflight,
            "required_stems": list(STEMS),
            "candidate_count": 0,
            "candidates": [],
            "windows": [],
            "verdict": "blocked_no_available_separator",
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "accepted_candidate_id": "",
            "presentation_status": "blocked_from_listen_pack",
            "listen_pack": {},
            "failed_checks": ["blocked_no_available_separator"],
            "unavailable_reason": recipe.get("unavailable_reason") or "no_default_separator_available",
        }
        _write_run_manifest_with_training_queue(output_dir, run_manifest)
        _append_jsonl(
            progress_path,
            {
                "event": "end",
                "verdict": "blocked_no_available_separator",
                "elapsed_sec": round(time.monotonic() - started, 6),
            },
        )
        return run_manifest

    if recipe is not None:
        try:
            windows = split_audio_windows(mix_path, output_dir, window_sec)
        except Exception as exc:
            run_manifest = {
                "schema": "fully_automatic_stem_search_run.v1",
                "mix_path": str(mix_path),
                "output_dir": str(output_dir),
                "window_sec": window_sec,
                "max_candidates": max_candidates,
                "gpu_required": require_gpu,
                "artifact_rag_write_allowed": not no_artifact_rag,
                "profile": profile or str((recipe or {}).get("profile") or "default"),
                "root_cause_route_policy": (recipe or {}).get("root_cause_route_policy") or {},
                "learned_route_policy": (recipe or {}).get("learned_route_policy") or {},
                "route_policy_checkpoint": str(route_policy_checkpoint or ""),
                "route_policy_mode": str(route_policy_mode or "shadow"),
                "route_budget_profile": str(route_budget_profile or ROUTE_BUDGET_PROFILE_FULL),
                "separator_backend": separator_backend_config.manifest(),
                "gpu_preflight": cuda_preflight,
                "required_stems": list(STEMS),
                "candidate_count": 0,
                "candidates": [],
                "windows": [],
                "verdict": "blocked_bad_reconstruction",
                "delivery_status": "blocked",
                "delivery_allowed": False,
                "accepted_candidate_id": "",
                "presentation_status": "blocked_from_listen_pack",
                "listen_pack": {},
                "failed_checks": ["blocked_bad_reconstruction"],
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            _write_run_manifest_with_training_queue(output_dir, run_manifest)
            _append_jsonl(
                progress_path,
                {"event": "end", "verdict": "blocked_bad_reconstruction", "elapsed_sec": round(time.monotonic() - started, 6)},
            )
            return run_manifest
        recipe_candidates = _candidate_items(recipe)[:max_candidates]
        runner = command_runner or default_command_runner
        timeout_sec = int(recipe.get("timeout_sec") or 300)
        recipe_model_manifest = str(recipe.get("contamination_model_manifest") or "").strip()
        recipe_profile = str(recipe.get("profile") or profile or "").strip()
        recipe_forensic_gate = _cross_stem_forensic_gate_config(recipe)
        for window in windows:
            window_dir = output_dir / str(window["window_id"])
            window_candidates: list[dict[str, Any]] = []
            target_specific_route_cache: dict[str, dict[str, Any]] = {}
            for idx, item in enumerate(recipe_candidates, start=1):
                item = dict(item)
                item.setdefault("candidate_id", f"c{idx:03d}")
                if recipe_model_manifest and not item.get("contamination_model_manifest"):
                    item["contamination_model_manifest"] = recipe_model_manifest
                if recipe_forensic_gate and not _cross_stem_forensic_gate_config(item):
                    item["cross_stem_forensic_gate"] = dict(recipe_forensic_gate)
                _append_jsonl(
                    progress_path,
                    {
                        "event": "candidate_start",
                        "window_id": window["window_id"],
                        "candidate_id": item["candidate_id"],
                        "elapsed_sec": round(time.monotonic() - started, 6),
                    },
                )
                candidate_manifest = execute_recipe_candidate(
                    candidate=item,
                    window=window,
                    window_dir=window_dir,
                    command_runner=runner,
                    timeout_sec=timeout_sec,
                    route_cache=target_specific_route_cache,
                    separator_backend_config=separator_backend_config,
                    route_budget_profile=route_budget_profile,
                )
                if learned_route_policy_model is not None:
                    candidate_manifest = _annotate_candidate_with_learned_route_policy(
                        candidate_manifest,
                        route_policy_model=learned_route_policy_model,
                        route_policy_mode=route_policy_mode,
                    )
                    _persist_candidate_manifest(candidate_manifest)
                window_candidates.append(candidate_manifest)
                candidates.append(candidate_manifest)
                _append_jsonl(
                    progress_path,
                    {
                        "event": "candidate_end",
                        "window_id": window["window_id"],
                        "candidate_id": item["candidate_id"],
                        "verdict": candidate_manifest["verdict"],
                        "elapsed_sec": round(time.monotonic() - started, 6),
                    },
                )
                if candidate_manifest["verdict"] == "auto_accept" and recipe_profile != MODEL_SPLIT_V2_PROFILE:
                    break
            if recipe_profile == MODEL_SPLIT_V2_PROFILE and not any(
                candidate.get("verdict") == "auto_accept" for candidate in window_candidates
            ):
                composite_template = dict(recipe_candidates[0]) if recipe_candidates else {"profile": MODEL_SPLIT_V2_PROFILE}
                if recipe_model_manifest and not composite_template.get("contamination_model_manifest"):
                    composite_template["contamination_model_manifest"] = recipe_model_manifest

                def _write_composite_progress(event: dict[str, Any]) -> None:
                    payload = dict(event)
                    payload["elapsed_sec"] = round(time.monotonic() - started, 6)
                    _append_jsonl(progress_path, payload)

                composite_candidates = build_model_split_v2_composite_candidates(
                    window=window,
                    window_dir=window_dir,
                    window_candidates=window_candidates,
                    candidate_template=composite_template,
                    max_composites=_effective_model_split_composite_limit(
                        recipe=recipe,
                        max_candidates=max_candidates,
                        default_recipe=recipe_path is None,
                    ),
                    progress_callback=_write_composite_progress,
                )
                for composite in composite_candidates:
                    if learned_route_policy_model is not None:
                        composite = _annotate_candidate_with_learned_route_policy(
                            composite,
                            route_policy_model=learned_route_policy_model,
                            route_policy_mode=route_policy_mode,
                        )
                        _persist_candidate_manifest(composite)
                    window_candidates.append(composite)
                    candidates.append(composite)
            window["candidates"] = window_candidates
    else:
        _append_jsonl(
            progress_path,
            {
                "event": "blocked",
                "reason": "missing_baseline_recipe",
                "verdict": "blocked_no_clean_candidate",
                "elapsed_sec": round(time.monotonic() - started, 6),
            },
        )

    selection = select_accepted_candidate(candidates, required_window_ids=[str(window["window_id"]) for window in windows])
    verdict = selection["verdict"]
    delivery_allowed = bool(selection["delivery_allowed"])
    presentation_status = "accepted_for_listen_pack" if verdict == "auto_accept" else "blocked_from_listen_pack"
    delivery_status = "accepted" if verdict == "auto_accept" else "blocked"
    failed_checks = [] if verdict == "auto_accept" else [verdict]
    listen_pack: dict[str, Any] = {}
    listen_pack_error: dict[str, str] | None = None
    listen_pack_forensic_score: dict[str, Any] | None = None
    if verdict == "auto_accept":
        try:
            listen_pack = build_listen_pack(
                output_dir=output_dir,
                mix_path=mix_path,
                windows=windows,
                accepted_window_candidates=selection.get("accepted_window_candidates") or {},
            )
            listen_forensic_config, listen_forensic_config_blockers = _resolve_listen_pack_forensic_config(
                recipe=recipe or {},
                windows=windows,
                accepted_window_candidates=selection.get("accepted_window_candidates") or {},
            )
            if listen_forensic_config_blockers:
                listen_pack_forensic_score = {
                    "schema": "cross_stem_forensic_matrix.v1",
                    "verdict": "blocked_unknown",
                    "blocker_reasons": listen_forensic_config_blockers,
                    "pair_results": [],
                    "config": listen_forensic_config,
                }
            else:
                listen_pack_forensic_score = validate_listen_pack_forensic(
                    listen_audio_paths=listen_pack.get("audio_paths") or {},
                    output_dir=output_dir / "listen_forensic",
                    config=listen_forensic_config,
                )
            if listen_pack_forensic_score.get("verdict") != "auto_accept":
                verdict = "blocked_unknown"
                delivery_allowed = False
                presentation_status = "blocked_from_listen_pack"
                delivery_status = "blocked"
                failed_checks = ["listen_pack_forensic_failed"]
                listen_pack = {}
                _remove_listen_pack(output_dir)
        except Exception as exc:
            verdict = "blocked_unknown"
            delivery_allowed = False
            presentation_status = "blocked_from_listen_pack"
            delivery_status = "blocked"
            failed_checks = ["listen_pack_build_failed"]
            listen_pack_error = {"type": type(exc).__name__, "message": str(exc)}
            _remove_listen_pack(output_dir)
    else:
        _remove_listen_pack(output_dir)
    run_manifest = {
        "schema": "fully_automatic_stem_search_run.v1",
        "mix_path": str(mix_path),
        "output_dir": str(output_dir),
        "window_sec": window_sec,
        "max_candidates": max_candidates,
        "gpu_required": require_gpu,
        "artifact_rag_write_allowed": not no_artifact_rag,
        "profile": profile or str((recipe or {}).get("profile") or "default"),
        "root_cause_route_policy": (recipe or {}).get("root_cause_route_policy") or {},
        "learned_route_policy": (recipe or {}).get("learned_route_policy") or {},
        "route_policy_checkpoint": str(route_policy_checkpoint or ""),
        "route_policy_mode": str(route_policy_mode or "shadow"),
        "route_budget_profile": str(route_budget_profile or ROUTE_BUDGET_PROFILE_FULL),
        "separator_backend": separator_backend_config.manifest(),
        "gpu_preflight": cuda_preflight,
        "required_stems": list(STEMS),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "windows": windows,
        "verdict": verdict,
        "delivery_status": delivery_status,
        "delivery_allowed": delivery_allowed,
        "accepted_candidate_id": selection["accepted_candidate_id"] if delivery_allowed else "",
        "accepted_window_candidates": selection.get("accepted_window_candidates") or {},
        "blocked_window_ids": selection.get("blocked_window_ids") or [],
        "presentation_status": presentation_status,
        "listen_pack": listen_pack,
        "failed_checks": failed_checks,
    }
    if listen_pack_error is not None:
        run_manifest["listen_pack_error"] = listen_pack_error
    if listen_pack_forensic_score is not None:
        run_manifest["listen_pack_forensic_score"] = listen_pack_forensic_score
    _write_run_manifest_with_training_queue(output_dir, run_manifest)
    _write_json(output_dir / "windows.json", {"schema": "fully_automatic_stem_windows.v1", "windows": windows})
    for window in windows:
        _append_jsonl(output_dir / "windows.jsonl", window)
    _append_jsonl(
        progress_path,
        {"event": "end", "verdict": verdict, "elapsed_sec": round(time.monotonic() - started, 6)},
    )
    return run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window-sec", type=float, default=6.0)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--require-gpu", default="true")
    parser.add_argument("--no-artifact-rag", action="store_true")
    parser.add_argument("--recipe")
    parser.add_argument("--profile", choices=("", MODEL_SPLIT_V2_PROFILE), default="")
    parser.add_argument("--contamination-model-manifest")
    parser.add_argument("--root-cause-queue", type=Path)
    parser.add_argument("--route-policy-checkpoint", type=Path)
    parser.add_argument("--route-policy-mode", choices=("shadow", "rank"), default="shadow")
    parser.add_argument("--route-budget-profile", choices=ROUTE_BUDGET_PROFILES, default=ROUTE_BUDGET_PROFILE_FULL)
    parser.add_argument("--separator-backend", choices=(MODE_SUBPROCESS, MODE_PERSISTENT_SERVER), default=MODE_SUBPROCESS)
    parser.add_argument("--separator-server-url", default="")
    parser.add_argument("--separator-device", default="")
    parser.add_argument("--separator-request-timeout-sec", type=float, default=0.0)
    parser.add_argument("--min-gpu-free-mib", type=int, default=DEFAULT_MIN_GPU_FREE_MIB)
    parser.add_argument("--wait-for-gpu-free-sec", type=float, default=0.0)
    parser.add_argument("--gpu-preflight-poll-sec", type=float, default=5.0)
    parser.add_argument(
        "--echo-progress",
        dest="echo_progress",
        action="store_true",
        default=True,
        help="Mirror progress.jsonl events to stderr during CLI runs",
    )
    parser.add_argument(
        "--no-echo-progress",
        dest="echo_progress",
        action="store_false",
        help="Write progress only to progress.jsonl",
    )
    parser.add_argument(
        "--stem-region-repair-checkpoint",
        action="append",
        default=[],
        help="Learned repair checkpoint spec: stem:issue_type:checkpoint_path or JSON object",
    )
    parser.add_argument(
        "--teacher-mask-checkpoint",
        action="append",
        default=[],
        help="Teacher-mask separator checkpoint candidate spec: role:stem:checkpoint_path",
    )
    parser.add_argument("--disable-default-recipe", action="store_true")
    return parser.parse_args()


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_stem_region_repair_checkpoint_spec(value: str, *, index: int = 1) -> dict[str, str]:
    raw_value = str(value or "").strip()
    if raw_value.startswith("@"):
        spec_path = Path(raw_value[1:])
        try:
            raw_value = spec_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise argparse.ArgumentTypeError(f"cannot read stem region repair checkpoint spec file: {spec_path}") from exc
    if raw_value.startswith("{"):
        try:
            raw = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(f"invalid stem region repair checkpoint JSON: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise argparse.ArgumentTypeError("stem region repair checkpoint JSON must be an object")
        checkpoint = str(raw.get("checkpoint") or raw.get("checkpoint_path") or "").strip()
        stem = str(raw.get("stem") or "").strip().lower()
        issue_type = str(raw.get("issue_type") or "").strip().lower()
        if not stem or not issue_type or not checkpoint:
            raise argparse.ArgumentTypeError("stem region repair checkpoint JSON requires stem, issue_type, checkpoint")
        result: dict[str, Any] = {
            "repair_id": _safe_artifact_segment(str(raw.get("repair_id") or f"sr{int(index):03d}"), fallback=f"sr{int(index):03d}"),
            "stem": stem,
            "issue_type": issue_type,
            "checkpoint": checkpoint,
        }
        for key in (
            "forensic_source_condition_stem",
            "mix_preserve_companion_stem",
            "delivery_pair_compensation_strength",
            "forensic_high_band_hz",
        ):
            raw_field = raw.get(key)
            if raw_field not in (None, ""):
                result[key] = raw_field
        region = raw.get("region")
        if isinstance(region, Mapping):
            result["region"] = {
                "start_sec": float(region.get("start_sec", 0.0)),
                "end_sec": float(region.get("end_sec", 0.0)),
            }
        return result
    parts = str(value or "").split(":", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "stem region repair checkpoint must be formatted as stem:issue_type:checkpoint_path"
        )
    stem, issue_type, checkpoint = (part.strip() for part in parts)
    return {
        "repair_id": f"sr{int(index):03d}",
        "stem": stem.lower(),
        "issue_type": issue_type.lower(),
        "checkpoint": checkpoint,
    }


def _parse_teacher_mask_checkpoint_spec(value: str, *, index: int = 1) -> dict[str, str]:
    parts = str(value or "").split(":", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "teacher-mask checkpoint must be formatted as role:stem:checkpoint_path"
        )
    role, stem, checkpoint = (part.strip() for part in parts)
    return {
        "route_id": f"tm{int(index):03d}_{role.strip().lower()}",
        "role": role.strip().lower(),
        "stem": stem.strip().lower(),
        "checkpoint": checkpoint,
    }


def main() -> None:
    args = parse_args()
    os.environ["STEM_SEARCH_ECHO_PROGRESS"] = "1" if args.echo_progress else "0"
    stem_region_repair_checkpoints = [
        _parse_stem_region_repair_checkpoint_spec(spec, index=index)
        for index, spec in enumerate(args.stem_region_repair_checkpoint or [], start=1)
    ]
    teacher_mask_checkpoints = [
        _parse_teacher_mask_checkpoint_spec(spec, index=index)
        for index, spec in enumerate(args.teacher_mask_checkpoint or [], start=1)
    ]
    result = run_fully_automatic_stem_search(
        mix_path=Path(args.mix),
        output_dir=Path(args.out),
        window_sec=args.window_sec,
        max_candidates=args.max_candidates,
        require_gpu=_parse_bool(args.require_gpu),
        no_artifact_rag=args.no_artifact_rag,
        recipe_path=Path(args.recipe) if args.recipe else None,
        allow_default_recipe=not args.disable_default_recipe,
        contamination_model_manifest=Path(args.contamination_model_manifest) if args.contamination_model_manifest else None,
        profile=args.profile,
        stem_region_repair_checkpoints=stem_region_repair_checkpoints,
        teacher_mask_checkpoints=teacher_mask_checkpoints,
        root_cause_queue_jsonl=args.root_cause_queue,
        route_policy_checkpoint=args.route_policy_checkpoint,
        route_policy_mode=args.route_policy_mode,
        separator_backend=args.separator_backend,
        separator_server_url=args.separator_server_url,
        separator_device=args.separator_device,
        separator_request_timeout_sec=args.separator_request_timeout_sec,
        min_gpu_free_mib=args.min_gpu_free_mib,
        wait_for_gpu_free_sec=args.wait_for_gpu_free_sec,
        gpu_preflight_poll_sec=args.gpu_preflight_poll_sec,
        route_budget_profile=args.route_budget_profile,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
