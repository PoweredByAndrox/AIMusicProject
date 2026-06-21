"""Run a fail-closed fully automatic six-stem search loop."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

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
TARGET_SPECIFIC_ROLE_OUTPUTS = {
    "six_stems": tuple(STEMS),
    "vocals": ("vocals",),
    "non_vocal_stems": tuple(NON_VOCAL_STEMS),
}
DEFAULT_VOCAL_ROUTE_ORDER = (
    "audio_separator_bs_roformer_vocals",
    "mdx23c_vocals",
    "melband_roformer_vocals",
    "becruily_instrumental_vocals",
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


def build_default_from_scratch_recipe(
    *,
    demucs_available: bool | None = None,
    available_vocal_routes: tuple[str, ...] | None = None,
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
    ) -> dict[str, Any]:
        candidate = {
            "candidate_id": candidate_id,
            "route_family": f"registered_six_stem_{model_id}",
            "registered_model_id": model_id,
            "conservation_repair_stem": "other",
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
            }
            candidate["vocal_projection_debleed"] = {
                "strength": debleed_strength,
                "target_stems": list(NON_VOCAL_STEMS),
            }
        return candidate

    return {
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
                "hybrid_htdemucs_ft_bs_roformer_6s",
                spec[5],
            )
            if spec[0] == "c010"
            else demucs_candidate(*spec)
            for spec in variant_specs
        ],
    }


def split_audio_windows(mix_path: Path, output_dir: Path, window_sec: float) -> list[dict[str, Any]]:
    audio, sample_rate = _read_audio(mix_path)
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")
    samples_per_window = max(1, int(round(window_sec * sample_rate)))
    windows: list[dict[str, Any]] = []
    for start in range(0, len(audio), samples_per_window):
        end = min(len(audio), start + samples_per_window)
        if end <= start:
            continue
        window_id = f"w{len(windows):03d}"
        window_path = _write_audio(output_dir / window_id / "mix.flac", audio[start:end], sample_rate)
        windows.append(
            {
                "window_id": window_id,
                "mix_path": str(window_path),
                "start_sec": round(start / sample_rate, 6),
                "end_sec": round(end / sample_rate, 6),
                "sample_rate": sample_rate,
                "sample_count": int(end - start),
                "channel_count": int(audio.shape[1]),
            }
        )
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
    return {stem: str(stems_dir / f"{stem}.flac") for stem in STEMS if (stems_dir / f"{stem}.flac").is_file()}


def _copy_target_specific_outputs(
    *,
    role: str,
    role_stems_dir: Path,
    final_stems_dir: Path,
    target_sample_rate: int | None = None,
    target_sample_count: int | None = None,
    target_channel_count: int | None = None,
) -> list[str]:
    required_outputs = TARGET_SPECIFIC_ROLE_OUTPUTS.get(role)
    if required_outputs is None:
        return [f"unsupported_target_specific_role:{role}"]
    blockers: list[str] = []
    for stem in required_outputs:
        source_path = role_stems_dir / f"{stem}.flac"
        if not source_path.is_file():
            blockers.append(f"missing_target_specific_output:{role}:{stem}")
            continue
        final_stems_dir.mkdir(parents=True, exist_ok=True)
        _copy_audio_to_match_shape(
            source_path,
            final_stems_dir / f"{stem}.flac",
            target_sample_rate=target_sample_rate,
            target_sample_count=target_sample_count,
            target_channel_count=target_channel_count,
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
) -> dict[str, Any]:
    routes: list[dict[str, Any]] = []
    blocker_reasons: list[str] = []
    for idx, item in enumerate(_target_specific_items(candidate), start=1):
        role = str(item.get("role") or "").strip()
        command_template = item.get("command_template")
        route_id = str(item.get("route_id") or role or f"r{idx:03d}")
        route_dir = candidate_dir / "routes" / route_id
        route_stems_dir = route_dir / "stems"
        route_manifest = {
            "role": role,
            "route_id": route_id,
            "output_dir": str(route_dir),
            "stems_dir": str(route_stems_dir),
            "required_outputs": list(TARGET_SPECIFIC_ROLE_OUTPUTS.get(role, ())),
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
        route_stems_dir.mkdir(parents=True, exist_ok=True)
        setattr(command_runner, "stems_dir", route_stems_dir)
        command_runner(command, timeout_sec)
        route_blockers = _copy_target_specific_outputs(
            role=role,
            role_stems_dir=route_stems_dir,
            final_stems_dir=stems_dir,
            target_sample_rate=int(window["sample_rate"]) if window.get("sample_rate") is not None else None,
            target_sample_count=int(window["sample_count"]) if window.get("sample_count") is not None else None,
            target_channel_count=int(window["channel_count"]) if window.get("channel_count") is not None else None,
        )
        route_manifest["missing_outputs"] = [
            reason.split(":")[-1] for reason in route_blockers if reason.startswith("missing_target_specific_output:")
        ]
        route_manifest["blocker_reasons"] = route_blockers
        blocker_reasons.extend(route_blockers)
        routes.append(route_manifest)
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


def validate_mix_conservation(*, mix_path: Path, stem_paths: Mapping[str, str]) -> dict[str, Any]:
    mix_audio, mix_rate = _read_audio(mix_path)
    summed = np.zeros_like(mix_audio, dtype=np.float32)
    blockers: list[str] = []
    for stem in STEMS:
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
    tolerance = max(1e-5, mix_rms * 0.01)
    if residual_rms > tolerance:
        blockers.append("mix_conservation_residual_too_high")
    return {
        "ok": not blockers,
        "blockers": blockers,
        "mix_rms": round(mix_rms, 8),
        "residual_rms": round(residual_rms, 8),
        "tolerance": round(tolerance, 8),
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


def apply_minus_vocal_debleed(
    *,
    stem_paths: Mapping[str, str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    low_hz = float(config.get("low_hz", 700.0))
    high_hz = float(config.get("high_hz", 1200.0))
    strength = float(config.get("strength", 1.0))
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
        stem_audios[stem] = stem_audio
    if blockers:
        return {
            "applied": False,
            "repair_owner": "runner_minus_vocal_debleed_v1",
            "reason": "debleed_inputs_invalid",
            "blockers": blockers,
        }

    for stem, stem_audio in stem_audios.items():
        moved = band_limited_audio(stem_audio, sample_rate, low_hz, high_hz) * np.float32(strength)
        moved_total = moved_total + moved
        per_stem[stem] = {"moved_rms": round(_rms(moved), 8)}

    moved_rms = _rms(moved_total)
    if moved_rms <= 1e-12:
        return {
            "applied": False,
            "repair_owner": "runner_minus_vocal_debleed_v1",
            "reason": "debleed_floor",
            "target_stems": target_stems,
            "moved_rms": 0.0,
        }
    for stem, stem_audio in stem_audios.items():
        moved = band_limited_audio(stem_audio, sample_rate, low_hz, high_hz) * np.float32(strength)
        _write_audio(Path(str(stem_paths[stem])), stem_audio.astype(np.float32) - moved, sample_rate)
    _write_audio(Path(str(vocals_path)), vocals_audio.astype(np.float32) + moved_total, sample_rate)
    return {
        "applied": True,
        "repair_owner": "runner_minus_vocal_debleed_v1",
        "low_hz": low_hz,
        "high_hz": high_hz,
        "strength": strength,
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
    mix_window_paths: list[Path] = []
    for window in windows:
        window_id = str(window["window_id"])
        candidate = _find_window_candidate(window, str(accepted_window_candidates[window_id]))
        stem_paths = candidate.get("stem_paths") if isinstance(candidate.get("stem_paths"), dict) else {}
        for stem in source_paths:
            stem_path = stem_paths.get(stem)
            if not stem_path:
                raise ValueError(f"listen pack missing stem path: {window_id}:{stem}")
            source_paths[stem].append(Path(str(stem_path)))
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


def execute_recipe_candidate(
    *,
    candidate: Mapping[str, Any],
    window: Mapping[str, Any],
    window_dir: Path,
    command_runner: CommandRunner,
    timeout_sec: int,
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

    minus_path = build_minus_from_non_vocal_stems({stem: Path(path) for stem, path in stem_paths.items()}, stems_dir / "minus.flac")
    stem_paths["minus"] = str(minus_path)
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
                "target_specific_composition": target_specific_composition,
                "next_action": next_action,
                "delivery_allowed": False,
                "presentation_status": "blocked_from_listen_pack",
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
                "target_specific_composition": target_specific_composition,
                "next_action": next_action,
            },
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
            "target_specific_composition": target_specific_composition,
            "contamination_score": score,
            "next_action": next_action,
            "delivery_allowed": score["verdict"] == "auto_accept",
            "presentation_status": "accepted_for_listen_pack" if score["verdict"] == "auto_accept" else "blocked_from_listen_pack",
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
        },
    )

    started = time.monotonic()
    candidates: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    recipe: dict[str, Any] | None = None
    if recipe_path is not None:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8-sig"))
    elif allow_default_recipe:
        recipe = build_default_from_scratch_recipe(demucs_available=demucs_available)

    if recipe is not None and recipe.get("available", True) is False:
        run_manifest = {
            "schema": "fully_automatic_stem_search_run.v1",
            "mix_path": str(mix_path),
            "output_dir": str(output_dir),
            "window_sec": window_sec,
            "max_candidates": max_candidates,
            "gpu_required": require_gpu,
            "artifact_rag_write_allowed": not no_artifact_rag,
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
        for window in windows:
            window_dir = output_dir / str(window["window_id"])
            window_candidates: list[dict[str, Any]] = []
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
                if candidate_manifest["verdict"] == "auto_accept":
                    break
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
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
