"""Run a fail-closed fully automatic six-stem search loop."""

from __future__ import annotations

import argparse
import json
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
from tools.stem_contamination_common import (
    FORBIDDEN_SOURCES,
    NON_VOCAL_STEMS,
    STEMS,
    VERDICTS,
    build_minus_from_non_vocal_stems,
    missing_required_stems,
    validate_no_artifact_rag_output,
)
from tools.stem_contamination_model import band_limited_audio


CommandRunner = Callable[[list[str], int], None]
DEFAULT_CONTAMINATION_MODEL_SEARCH_ROOTS = (
    Path("/project_artifacts/stem_eval"),
    Path("/host_artifacts/stem_eval"),
    Path("artifacts/stem_eval"),
)
TARGET_SPECIFIC_ROLE_OUTPUTS = {
    "six_stems": tuple(STEMS),
    "vocals": ("vocals",),
    "minus": ("minus",),
    "bass_support": ("bass",),
    "four_stem_non_vocal": ("drums", "bass", "other"),
    "non_vocal_stems": tuple(NON_VOCAL_STEMS),
}
MODEL_SPLIT_V2_PROFILE = "model_split_v2"
DEFAULT_VOCAL_ROUTE_ORDER = (
    "audio_separator_bs_roformer_vocals",
    "mdx23c_vocals",
    "melband_roformer_vocals",
    "becruily_instrumental_vocals",
)
MODEL_SPLIT_V2_VOCAL_LANE = (
    "audio_separator_bs_roformer_vocals",
    "mdx23c_vocals",
    "melband_roformer_vocals",
)
MODEL_SPLIT_V2_MINUS_LANE = (
    "becruily_instrumental_vocals",
    "audio_separator_bs_roformer_vocals:instrumental",
    "mdx23c_vocals:instrumental",
    "htdemucs_ft:instrumental",
)
MODEL_SPLIT_V2_STEM_LANE = (
    "htdemucs_6s",
    "hybrid_htdemucs_ft_bs_roformer_6s",
    "hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first",
)
MODEL_SPLIT_V2_BASS_SUPPORT_LANE = (
    "scnet_xl_ihf_musdb18:bass",
)
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
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


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


def _minus_lane_command_template(route_id: str) -> list[str]:
    base_route = route_id.split(":", 1)[0]
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


def build_model_split_v2_recipe(
    *,
    demucs_available: bool | None = None,
    available_vocal_routes: tuple[str, ...] | None = None,
    contamination_model_manifest: Path | str | None = None,
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

    vocal_lane = tuple(
        route for route in MODEL_SPLIT_V2_VOCAL_LANE if available_vocal_routes is None or route in set(available_vocal_routes)
    )
    if available_vocal_routes is not None:
        # Keep the contract-visible lane list stable even when callers restrict
        # the first executable routes in tests or diagnostics.
        vocal_lane_manifest = tuple(MODEL_SPLIT_V2_VOCAL_LANE)
    else:
        vocal_lane_manifest = tuple(MODEL_SPLIT_V2_VOCAL_LANE)
    executable_vocal_lane = vocal_lane or tuple(MODEL_SPLIT_V2_VOCAL_LANE)
    candidates: list[dict[str, Any]] = []
    for index in range(10):
        stem_route = MODEL_SPLIT_V2_STEM_LANE[index % len(MODEL_SPLIT_V2_STEM_LANE)]
        vocal_route = executable_vocal_lane[index % len(executable_vocal_lane)]
        minus_route = MODEL_SPLIT_V2_MINUS_LANE[index % len(MODEL_SPLIT_V2_MINUS_LANE)]
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
                "command_template": _vocal_route_command_template(vocal_route),
            },
            {
                "role": "minus",
                "lane": "minus_lane",
                "route_id": minus_route,
                "output_map": minus_output_map,
                "command_template": _minus_lane_command_template(minus_route),
            },
        ]
        if index == 0:
            bass_route = MODEL_SPLIT_V2_BASS_SUPPORT_LANE[0]
            target_specific_commands.append(
                {
                    "role": "bass_support",
                    "lane": "bass_support_lane",
                    "route_id": bass_route,
                    "command_template": _bass_support_lane_command_template(bass_route),
                }
            )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "profile": MODEL_SPLIT_V2_PROFILE,
                "route_family": f"model_split_v2_{stem_route}_{vocal_route}_{minus_route.replace(':', '_')}",
                "required_stems": list(STEMS),
                "minus_source": "minus_lane",
                "minus_rebuilt_from_non_vocal_stems": False,
                "model_split_lanes": {
                    "vocals": "vocal_lane",
                    "minus": "minus_lane",
                    "stems": "stem_lane",
                    "bass_support": "bass_support_lane",
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
            "minus_lane": list(MODEL_SPLIT_V2_MINUS_LANE),
            "stem_lane": list(MODEL_SPLIT_V2_STEM_LANE),
            "bass_support_lane": list(MODEL_SPLIT_V2_BASS_SUPPORT_LANE),
            "auxiliary_non_vocal_evidence": ["scnet_xl_ihf_musdb18"],
        },
        "model_split_composite_max_candidates": 72,
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
        "minus_rebuilt_from_non_vocal_stems": True,
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
    }


def _candidate_items(recipe: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items = recipe.get("candidates")
    if not isinstance(items, list):
        items = recipe.get("commands")
    return [item for item in (items or []) if isinstance(item, dict)]


def _target_specific_items(candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items = candidate.get("target_specific_commands")
    return [item for item in (items or []) if isinstance(item, dict)] if isinstance(items, list) else []


def _collect_stem_paths(stems_dir: Path) -> dict[str, str]:
    return {stem: str(stems_dir / f"{stem}.flac") for stem in [*STEMS, "minus"] if (stems_dir / f"{stem}.flac").is_file()}


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
) -> dict[str, Any]:
    routes: list[dict[str, Any]] = []
    blocker_reasons: list[str] = []
    for idx, item in enumerate(_target_specific_items(candidate), start=1):
        role = str(item.get("role") or "").strip()
        command_template = item.get("command_template")
        route_id = str(item.get("route_id") or role or f"r{idx:03d}")
        route_alias = _safe_artifact_segment(route_id, fallback=f"r{idx:03d}")
        lane = str(item.get("lane") or role or route_id).strip()
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
        cache_key = json.dumps(
            {
                "role": role,
                "lane": lane,
                "route_id": route_id,
                "command_template": command_template,
                "output_map": dict(output_map),
            },
            sort_keys=True,
        )
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
                }
            )
            blocker_reasons.extend(route_blockers)
            routes.append(route_manifest)
            continue
        route_stems_dir.mkdir(parents=True, exist_ok=True)
        setattr(command_runner, "stems_dir", route_stems_dir)
        command_runner(command, timeout_sec)
        copied_outputs: list[dict[str, str]] = []
        route_blockers = _copy_target_specific_outputs(
            role=role,
            role_stems_dir=route_stems_dir,
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
                "stems_dir": str(route_stems_dir),
                "route_id": route_id,
                "role": role,
                "lane": lane,
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
    default_model_manifest = str(candidate.get("contamination_model_manifest") or "").strip()
    checks = candidate.get("contamination_checks")
    if not isinstance(checks, list):
        if not default_model_manifest:
            return required_checks, False
        return [{**check, "model_manifest": default_model_manifest} for check in required_checks], True

    hints: dict[tuple[str, str], dict[str, str]] = {}
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
    invalid_targets = [stem for stem in target_stems if stem not in NON_VOCAL_STEMS]
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
        per_stem[stem] = {
            "moved_rms": round(_rms(moved_by_stem[stem]), 8),
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
    invalid_targets = [stem for stem in target_stems if stem not in NON_VOCAL_STEMS]
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


def _find_window_candidate(window: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    for candidate in window.get("candidates", []):
        if str(candidate.get("candidate_id") or "") == candidate_id and candidate.get("verdict") == "auto_accept":
            return candidate
    raise ValueError(f"accepted candidate not found for window {window.get('window_id')}: {candidate_id}")


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
    return str(candidate.get("minus_source") or ("minus_lane" if _is_model_split_v2(candidate) else "rebuilt_non_vocal_stems"))


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
        "check_mode": "full_forbidden_matrix",
        "candidate_check_hints_used": explicit_checks,
        "required_check_count": len(checks),
    }


def _route_source_for_stem(route: Mapping[str, Any], stem: str) -> Path | None:
    for copied in route.get("copied_outputs", []) if isinstance(route.get("copied_outputs"), list) else []:
        if not isinstance(copied, dict):
            continue
        if str(copied.get("stem") or "") != stem:
            continue
        source_path = Path(str(copied.get("source_path") or ""))
        return source_path if source_path.is_file() else None
    return None


def _route_has_source_stems(route: Mapping[str, Any], stems: Sequence[str]) -> bool:
    return all(_route_source_for_stem(route, stem) is not None for stem in stems)


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
            key = str(route.get("route_id") or route.get("route_alias") or route.get("stems_dir") or "")
            if key in seen:
                continue
            seen.add(key)
            routes.append(route)
    return routes


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
) -> dict[str, Any]:
    verdict = str(
        (contamination_score or voice_presence_score or damage_score or reconstruction).get("verdict")
        if any((contamination_score, voice_presence_score, damage_score))
        else ("auto_accept" if reconstruction.get("ok") else "blocked_bad_reconstruction")
    )
    if not reconstruction.get("ok"):
        verdict = "blocked_bad_reconstruction"
    next_action = choose_next_action(
        {
            "verdict": verdict,
            "reason": ",".join(
                list((contamination_score or {}).get("blocker_reasons") or [])
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
            "profile": MODEL_SPLIT_V2_PROFILE,
            "minus_source": "minus_lane",
            "minus_rebuilt_from_non_vocal_stems": False,
            "stems_dir": str(candidate_dir / "stems"),
            "score_path": str(candidate_dir / "score.json"),
            "required_stems": list(STEMS),
            "stem_paths": dict(stem_paths),
            "missing_stems": [],
            "reconstruction": reconstruction,
            "target_specific_composition": target_specific_composition,
            "damage_score": damage_score,
            "voice_presence_score": voice_presence_score,
            "contamination_score": contamination_score,
            "next_action": next_action,
            "verdict": verdict,
            "delivery_allowed": verdict == "auto_accept",
            "presentation_status": "accepted_for_listen_pack" if verdict == "auto_accept" else "blocked_from_listen_pack",
        }
    )
    score_payload = {
        "verdict": verdict,
        "reconstruction": reconstruction,
        "damage_score": damage_score,
        "voice_presence_score": voice_presence_score,
        "contamination_score": contamination_score,
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
    max_composites: int = 72,
) -> list[dict[str, Any]]:
    stem_routes = _model_split_routes_by_role(window_candidates, "six_stems", NON_VOCAL_STEMS)
    vocal_routes = _model_split_routes_by_role(window_candidates, "vocals", ("vocals",))
    minus_routes = _model_split_routes_by_role(window_candidates, "minus", ("minus",))
    bass_support_routes = _model_split_routes_by_role(window_candidates, "bass_support", ("bass",))
    bass_route_options: list[Mapping[str, Any] | None] = [None, *bass_support_routes]
    composite_candidates: list[dict[str, Any]] = []
    composite_index = 0
    for stem_route in stem_routes:
        for vocal_route in vocal_routes:
            for minus_route in minus_routes:
                for bass_route in bass_route_options:
                    composite_index += 1
                    if composite_index > max_composites:
                        return composite_candidates
                    candidate_id = f"c{100 + composite_index:03d}"
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
                            "bass_support_applied": bass_route is not None,
                        }
                    )
                    stem_copies: list[dict[str, str]] = []
                    vocal_copies: list[dict[str, str]] = []
                    minus_copies: list[dict[str, str]] = []
                    bass_copies: list[dict[str, str]] = []
                    try:
                        for stem in NON_VOCAL_STEMS:
                            _copy_route_stem_to_candidate(
                                route=stem_route,
                                stem=stem,
                                final_stems_dir=stems_dir,
                                window=window,
                                copied_outputs=stem_copies,
                            )
                        if bass_route is not None:
                            _copy_route_stem_to_candidate(
                                route=bass_route,
                                stem="bass",
                                final_stems_dir=stems_dir,
                                window=window,
                                copied_outputs=bass_copies,
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
                        composite_candidates.append(manifest)
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
                    if bass_route is not None:
                        routes.append(
                            _composite_route_manifest(
                                source_route=bass_route,
                                role="bass_support",
                                lane="bass_support_lane",
                                copied_outputs=bass_copies,
                            )
                        )
                    target_specific_composition = {
                        "applied": True,
                        "composite_lane_selection": True,
                        "bass_support_applied": bass_route is not None,
                        "blocker_reasons": [],
                        "routes": routes,
                    }
                    stem_paths = _collect_stem_paths(stems_dir)
                    missing = missing_required_stems(stem_paths) + ([] if stem_paths.get("minus") else ["minus"])
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
                        composite_candidates.append(manifest)
                        continue

                    reconstruction_config = candidate.get("delivery_pair_reconstruction")
                    reconstruction = validate_model_split_v2_conservation(
                        mix_path=Path(str(window["mix_path"])),
                        stem_paths=stem_paths,
                        target_specific_composition=target_specific_composition,
                        config=reconstruction_config if isinstance(reconstruction_config, Mapping) else None,
                    )
                    if not reconstruction["ok"]:
                        composite_candidates.append(
                            _finalize_composite_candidate(
                                candidate=candidate,
                                candidate_dir=candidate_dir,
                                stem_paths=stem_paths,
                                target_specific_composition=target_specific_composition,
                                reconstruction=reconstruction,
                            )
                        )
                        continue
                    damage_score = score_model_split_v2_damage_gates(
                        candidate=candidate,
                        mix_path=Path(str(window["mix_path"])),
                        stem_paths=stem_paths,
                    )
                    if damage_score["verdict"] != "auto_accept":
                        composite_candidates.append(
                            _finalize_composite_candidate(
                                candidate=candidate,
                                candidate_dir=candidate_dir,
                                stem_paths=stem_paths,
                                target_specific_composition=target_specific_composition,
                                reconstruction=reconstruction,
                                damage_score=damage_score,
                            )
                        )
                        continue
                    voice_presence_score = score_model_split_v2_voice_presence_gate(candidate=candidate, stem_paths=stem_paths)
                    if voice_presence_score["verdict"] != "auto_accept":
                        composite_candidates.append(
                            _finalize_composite_candidate(
                                candidate=candidate,
                                candidate_dir=candidate_dir,
                                stem_paths=stem_paths,
                                target_specific_composition=target_specific_composition,
                                reconstruction=reconstruction,
                                damage_score=damage_score,
                                voice_presence_score=voice_presence_score,
                            )
                        )
                        continue
                    contamination_score = _score_candidate_contamination(
                        candidate=candidate,
                        candidate_dir=candidate_dir,
                        stem_paths=stem_paths,
                    )
                    composite_candidates.append(
                        _finalize_composite_candidate(
                            candidate=candidate,
                            candidate_dir=candidate_dir,
                            stem_paths=stem_paths,
                            target_specific_composition=target_specific_composition,
                            reconstruction=reconstruction,
                            damage_score=damage_score,
                            voice_presence_score=voice_presence_score,
                            contamination_score=contamination_score,
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
            "minus_rebuilt_from_non_vocal_stems": bool(candidate.get("minus_rebuilt_from_non_vocal_stems", True)),
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
    try:
        if target_specific_commands:
            target_specific_composition = _run_target_specific_commands(
                candidate=candidate,
                window=window,
                candidate_dir=candidate_dir,
                stems_dir=stems_dir,
                command_runner=command_runner,
                timeout_sec=timeout_sec,
                route_cache=route_cache,
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
                "missing_stems": missing_required_stems(stem_paths),
                "stem_paths": stem_paths,
                "target_specific_composition": target_specific_composition,
                "next_action": next_action,
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
    missing = missing_required_stems(stem_paths)
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

    minus_rebuilt_from_non_vocal_stems = bool(candidate.get("minus_rebuilt_from_non_vocal_stems", True))
    if _is_model_split_v2(candidate) or str(candidate.get("minus_source") or "") == "minus_lane":
        minus_rebuilt_from_non_vocal_stems = False
    if minus_rebuilt_from_non_vocal_stems:
        minus_path = build_minus_from_non_vocal_stems({stem: Path(path) for stem, path in stem_paths.items()}, stems_dir / "minus.flac")
        stem_paths["minus"] = str(minus_path)
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
        reconstruction = validate_mix_conservation(mix_path=Path(str(window["mix_path"])), stem_paths=stem_paths)
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
    next_action = choose_next_action(
        {
            "verdict": score["verdict"],
            "reason": ",".join(score.get("blocker_reasons") or []),
        }
    )
    manifest.update(
        {
            "execution_status": "completed",
            "candidate_untrusted": score["verdict"] != "auto_accept",
            "verdict": score["verdict"],
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
            "contamination_score": score,
            "next_action": next_action,
            "delivery_allowed": score["verdict"] == "auto_accept",
            "presentation_status": "accepted_for_listen_pack" if score["verdict"] == "auto_accept" else "blocked_from_listen_pack",
            "minus_source": _candidate_minus_source(candidate),
            "minus_rebuilt_from_non_vocal_stems": minus_rebuilt_from_non_vocal_stems,
        }
    )
    score["next_action"] = next_action
    _write_json(candidate_dir / "score.json", score)
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
                    if str(candidate.get("window_id") or "") == window_id and candidate.get("verdict") == "auto_accept"
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
        if candidate.get("verdict") == "auto_accept":
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
) -> dict[str, Any]:
    if no_artifact_rag:
        validate_no_artifact_rag_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
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
        },
    )

    started = time.monotonic()
    candidates: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    recipe: dict[str, Any] | None = None
    if recipe_path is not None:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8-sig"))
    elif allow_default_recipe:
        model_manifest = contamination_model_manifest or _discover_default_contamination_model_manifest()
        if profile == MODEL_SPLIT_V2_PROFILE:
            recipe = build_model_split_v2_recipe(
                demucs_available=demucs_available,
                contamination_model_manifest=model_manifest,
            )
        else:
            recipe = build_default_from_scratch_recipe(
                demucs_available=demucs_available,
                contamination_model_manifest=model_manifest,
            )
    if recipe is not None and contamination_model_manifest is not None and not recipe.get("contamination_model_manifest"):
        recipe["contamination_model_manifest"] = str(contamination_model_manifest)

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
        _write_json(output_dir / "run.json", run_manifest)
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
            _write_json(output_dir / "run.json", run_manifest)
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
        for window in windows:
            window_dir = output_dir / str(window["window_id"])
            window_candidates: list[dict[str, Any]] = []
            target_specific_route_cache: dict[str, dict[str, Any]] = {}
            for idx, item in enumerate(recipe_candidates, start=1):
                item = dict(item)
                item.setdefault("candidate_id", f"c{idx:03d}")
                if recipe_model_manifest and not item.get("contamination_model_manifest"):
                    item["contamination_model_manifest"] = recipe_model_manifest
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
                )
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
                composite_candidates = build_model_split_v2_composite_candidates(
                    window=window,
                    window_dir=window_dir,
                    window_candidates=window_candidates,
                    candidate_template=composite_template,
                    max_composites=int(recipe.get("model_split_composite_max_candidates") or 72),
                )
                for composite in composite_candidates:
                    window_candidates.append(composite)
                    candidates.append(composite)
                    _append_jsonl(
                        progress_path,
                        {
                            "event": "composite_candidate_end",
                            "window_id": window["window_id"],
                            "candidate_id": composite.get("candidate_id"),
                            "verdict": composite.get("verdict"),
                            "elapsed_sec": round(time.monotonic() - started, 6),
                        },
                    )
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
    if verdict == "auto_accept":
        try:
            listen_pack = build_listen_pack(
                output_dir=output_dir,
                mix_path=mix_path,
                windows=windows,
                accepted_window_candidates=selection.get("accepted_window_candidates") or {},
            )
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
    _write_json(output_dir / "run.json", run_manifest)
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
    parser.add_argument("--disable-default-recipe", action="store_true")
    return parser.parse_args()


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    args = parse_args()
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
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
