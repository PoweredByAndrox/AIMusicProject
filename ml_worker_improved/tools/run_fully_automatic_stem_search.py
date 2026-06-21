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


CommandRunner = Callable[[list[str], int], None]


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


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


def _render_template(value: str, replacements: Mapping[str, str]) -> str:
    rendered = value
    for key, replacement in replacements.items():
        rendered = rendered.replace("{" + key + "}", replacement)
    return rendered


def render_recipe_command(command_template: list[Any], replacements: Mapping[str, str]) -> list[str]:
    return [_render_template(str(part), replacements) for part in command_template]


def default_command_runner(command: list[str], timeout_sec: int) -> None:
    subprocess.run(command, check=True, timeout=timeout_sec)


def build_default_from_scratch_recipe(*, demucs_available: bool | None = None) -> dict[str, Any]:
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
    variant_specs = (
        ("c001", "6", "0.25", "1", ""),
        ("c002", "6", "0.50", "1", ""),
        ("c003", "6", "0.25", "2", ""),
        ("c004", "5", "0.50", "1", ""),
        ("c005", "6", "0.25", "1", "other"),
    )

    def demucs_candidate(candidate_id: str, segment: str, overlap: str, shifts: str, residual_stem: str) -> dict[str, Any]:
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
        return {
            "candidate_id": candidate_id,
            "route_family": "demucs_htdemucs_6s",
            "residual_correction_stem": residual_stem,
            "conservation_repair_stem": "other",
            "required_stems": list(STEMS),
            "contamination_checks": [{"tested_stem": "minus", "foreign_source": "vocals"}],
            "command_template": command_template,
        }

    return {
        "schema": "stem_search_recipe.v1",
        "recipe_source": "from_scratch_default",
        "available": True,
        "timeout_sec": 900,
        "candidates": [demucs_candidate(*spec) for spec in variant_specs],
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


def _collect_stem_paths(stems_dir: Path) -> dict[str, str]:
    return {stem: str(stems_dir / f"{stem}.flac") for stem in STEMS if (stems_dir / f"{stem}.flac").is_file()}


def _candidate_contamination_checks(candidate: Mapping[str, Any]) -> tuple[list[dict[str, str]], bool]:
    checks = candidate.get("contamination_checks")
    if isinstance(checks, list):
        normalized: list[dict[str, str]] = []
        for item in checks:
            if not isinstance(item, dict):
                continue
            tested_stem = str(item.get("tested_stem") or "").strip()
            foreign_source = str(item.get("foreign_source") or "").strip()
            if tested_stem and foreign_source:
                check = {"tested_stem": tested_stem, "foreign_source": foreign_source}
                model_manifest = str(item.get("model_manifest") or "").strip()
                if model_manifest:
                    check["model_manifest"] = model_manifest
                normalized.append(check)
        return normalized, True

    return [
        {"tested_stem": tested_stem, "foreign_source": foreign_source}
        for tested_stem, forbidden_sources in FORBIDDEN_SOURCES.items()
        for foreign_source in forbidden_sources
    ], False


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
    residual_rms = _rms(residual)
    if residual_rms <= 1e-12:
        return {
            "applied": False,
            "repair_owner": "runner_conservation_repair_v1",
            "reason": "residual_floor",
            "target_stem": target_stem,
            "pre_repair_residual_rms": 0.0,
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

    if explicit_checks and not checks:
        blocker_reasons.append("empty_contamination_checks")

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
            blocker_reasons.append(f"contamination_detector_unknown:{tested_stem}:{foreign_source}")

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
        "check_mode": "explicit_candidate_checks" if explicit_checks else "full_forbidden_matrix",
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
    if not isinstance(command_template, list) or not command_template:
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
    stems_dir.mkdir(parents=True, exist_ok=True)
    setattr(command_runner, "stems_dir", stems_dir)
    try:
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
    missing = missing_required_stems(stem_paths)
    if missing:
        next_action = choose_next_action({"verdict": "blocked_missing_stem", "reason": "missing_required_stem"})
        manifest.update(
            {
                "execution_status": "completed",
                "verdict": "blocked_missing_stem",
                "missing_stems": missing,
                "stem_paths": stem_paths,
                "next_action": next_action,
            }
        )
        _write_json(candidate_dir / "manifest.json", manifest)
        _write_json(candidate_dir / "score.json", {"verdict": "blocked_missing_stem", "missing_stems": missing, "next_action": next_action})
        return manifest

    repair_stem = str(candidate.get("conservation_repair_stem") or "").strip()
    reconstruction_repair: dict[str, Any] | None = None
    if repair_stem:
        reconstruction_repair = apply_mix_conservation_repair(
            mix_path=Path(str(window["mix_path"])),
            stem_paths=stem_paths,
            target_stem=repair_stem,
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
            "delivery_allowed": False,
            "accepted_candidate_id": "",
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
                "delivery_allowed": False,
                "accepted_candidate_id": "",
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
        for window in windows:
            window_dir = output_dir / str(window["window_id"])
            window_candidates: list[dict[str, Any]] = []
            for idx, item in enumerate(recipe_candidates, start=1):
                item = dict(item)
                item.setdefault("candidate_id", f"c{idx:03d}")
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
        "delivery_allowed": bool(selection["delivery_allowed"]),
        "accepted_candidate_id": selection["accepted_candidate_id"],
        "accepted_window_candidates": selection.get("accepted_window_candidates") or {},
        "blocked_window_ids": selection.get("blocked_window_ids") or [],
        "presentation_status": "accepted_for_listen_pack" if verdict == "auto_accept" else "blocked_from_listen_pack",
        "failed_checks": [] if verdict == "auto_accept" else [verdict],
    }
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
