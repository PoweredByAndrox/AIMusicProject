from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from worker.model_registry import ModelRegistry, ModelRegistryEntry, default_model_registry  # noqa: E402


DEFAULT_TARGET_STEMS = ("vocal", "minus", "melody", "drums", "bass", "other")
DEFAULT_DOCKER_SOURCE_ROOT = "/host_testing_stuff_music"
DEFAULT_MAX_MODELS_PER_STEM = 50
EXPECTED_ACCELERATION_MODE = "gpu_inference_then_cpu_dsp_scoring"
CPU_DSP_ACCELERATION_MODE = "cpu_dsp_candidate_generation"
SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "audio_copied": False,
    "audio_exported": False,
    "audio_hashed": False,
    "release_ready": False,
    "execution_started": False,
}
MINUS_BASELINE_REQUIRED_PATHS = (
    "/app/vendor/bs_roformer_instrumental_resurrection_manual/config_bs_roformer_instrumental_resurrection_unwa.yaml",
    "/app/vendor/bs_roformer_instrumental_resurrection_manual/bs_roformer_instrumental_resurrection_unwa.ckpt",
    "/app/vendor/music_source_separation_training",
)
STEM_LANE_MINUS_EXPORTERS = (
    "export_hq_stems_full.py",
    "run_registered_non_vocal_candidate.py",
)


def stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")


def safe_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:96] or "unknown"


def normalize_stem(stem: str) -> str:
    value = str(stem or "").strip().lower()
    if value in {"vocals", "voice"}:
        return "vocal"
    if value in {"instrumental", "instrumentals"}:
        return "minus"
    return value


def registry_stem_name(stem: str) -> str:
    stem = normalize_stem(stem)
    if stem == "vocal":
        return "vocals"
    if stem == "minus":
        return "instrumental"
    return stem


def entry_known_labels(entry: ModelRegistryEntry) -> set[str]:
    return {
        entry.model_id.lower(),
        entry.family.lower(),
        *(alias.lower() for alias in entry.aliases),
    }


def registry_entry_suggestion_dict(
    *,
    registry: ModelRegistry,
    entry: ModelRegistryEntry,
    reason: str,
) -> dict[str, Any]:
    availability = registry.availability(entry)
    return {
        "model_id": entry.model_id,
        "display_name": entry.display_name,
        "family": entry.family,
        "target_stems": list(entry.target_stems),
        "output_stems": list(entry.output_stems),
        "availability": {
            "status": availability.status,
            "missing_paths": list(availability.missing_paths),
            "missing_python_modules": list(availability.missing_python_modules),
            "runtime_blockers": list(getattr(availability, "runtime_blockers", ())),
        },
        "reason": reason,
        "command_template": list(entry.command_template),
        "expected_manifest": entry.expected_manifest,
        "task_scope": entry.task_scope,
        "benchmark_score_context": entry.benchmark_score_context,
        "license_status": entry.license_status,
        "code_license": entry.code_license,
        "weights_license": entry.weights_license,
        "commercial_use_status": entry.commercial_use_status,
        "canonical_name_status": entry.canonical_name_status,
        "comparability_notes": list(entry.comparability_notes),
        "report_alignment_notes": list(entry.report_alignment_notes),
    }


def suggestion_dict_sort_key(suggestion: dict[str, Any]) -> tuple[int, str, str]:
    availability = suggestion.get("availability") if isinstance(suggestion.get("availability"), dict) else {}
    available_rank = 0 if availability.get("status") == "available" else 1
    return (available_rank, str(suggestion.get("family") or ""), str(suggestion.get("model_id") or ""))


def row_stems(row: dict[str, Any], requested_stems: Sequence[str] | None) -> list[str]:
    if requested_stems:
        candidates = requested_stems
    else:
        candidates = row.get("target_stems") or DEFAULT_TARGET_STEMS
    normalized = [normalize_stem(str(stem)) for stem in candidates]
    return [stem for stem in DEFAULT_TARGET_STEMS if stem in normalized]


def source_audio_path(row: dict[str, Any]) -> str:
    return str(row.get("docker_audio_path") or row.get("source_audio_path") or "").strip()


def path_exists(path: str) -> bool:
    return bool(path) and Path(path).exists()


def minus_baseline_availability() -> dict[str, Any]:
    missing_paths = [path for path in MINUS_BASELINE_REQUIRED_PATHS if not Path(path).exists()]
    return {
        "status": "available" if not missing_paths else "unavailable",
        "missing_paths": missing_paths,
        "missing_python_modules": [],
    }


def downstream_gates(stem: str) -> list[str]:
    gates = [
        "candidate_manifest_contract",
        "mode_specific_score_matrix",
        "artifact_status_contract",
    ]
    if stem in {"vocal", "minus"}:
        gates.extend(
            [
                "no_reference_windowed_judge",
                "mode_specific_optimizer",
            ]
        )
    if stem == "minus":
        gates.append("minus_musical_detail_preservation_gate")
    if stem in {"melody", "drums", "bass", "other"}:
        gates.append("stem_quality_control")
    return gates


def job_status_for(*, source_exists: bool, availability_status: str) -> str:
    if not source_exists:
        return "blocked_missing_source_audio"
    if availability_status != "available":
        return "blocked_model_unavailable"
    return "queued"


def acceleration_contract_for_route_family(route_family: str) -> dict[str, Any]:
    normalized_family = str(route_family or "").strip().lower()
    if normalized_family == "dsp":
        return {
            "expected_acceleration_mode": CPU_DSP_ACCELERATION_MODE,
            "gpu_inference_required": False,
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "cpu_only_dsp_route",
        }
    return {
        "expected_acceleration_mode": EXPECTED_ACCELERATION_MODE,
        "gpu_inference_required": True,
        "gpu_inference_skipped": False,
        "gpu_inference_skip_reason": "",
    }


def load_route_prior_index(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    priors: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        stem_type = normalize_stem(str(row.get("stem_type") or ""))
        route_id = str(row.get("route_id") or "").strip()
        if not stem_type or not route_id:
            continue
        priors[(stem_type, route_id)] = row
    return priors


def route_prior_blocks_job(prior: dict[str, Any]) -> bool:
    recommendation = str(prior.get("recommendation") or "")
    return bool(prior.get("requires_new_source_candidate") is True or recommendation == "blocked_as_minus_baseline")


def route_prior_candidates_for_job(row: dict[str, Any]) -> list[tuple[str, str]]:
    stem_type = normalize_stem(str(row.get("target_stem") or ""))
    return [
        (stem_type, route_id)
        for route_id in [
            str(row.get("route_name") or "").strip(),
            str(row.get("model_id") or "").strip(),
            str(row.get("route_family") or "").strip(),
        ]
        if route_id
    ]


def matching_blocking_route_prior(
    row: dict[str, Any],
    route_priors: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    for key in route_prior_candidates_for_job(row):
        prior = route_priors.get(key)
        if prior and route_prior_blocks_job(prior):
            return prior
    return None


def source_pool_route_priors(
    route_priors: dict[tuple[str, str], dict[str, Any]],
    stem: str,
) -> list[dict[str, Any]]:
    stem_type = normalize_stem(stem)
    rows = [
        prior
        for (prior_stem_type, _route_id), prior in route_priors.items()
        if prior_stem_type == stem_type and prior.get("source_pool_eligible") is True
    ]
    return sorted(
        rows,
        key=lambda prior: (
            str(prior.get("best_source_track_id") or ""),
            str(prior.get("route_id") or ""),
            str(prior.get("best_source_candidate_id") or ""),
        ),
    )


def blocked_route_ids_for_stem(
    route_priors: dict[tuple[str, str], dict[str, Any]],
    stem: str,
) -> set[str]:
    stem_type = normalize_stem(stem)
    return {
        route_id
        for (prior_stem_type, route_id), prior in route_priors.items()
        if prior_stem_type == stem_type and route_prior_blocks_job(prior)
    }


def apply_route_prior_policy(
    row: dict[str, Any],
    route_priors: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    prior = matching_blocking_route_prior(row, route_priors)
    if prior is None:
        return row
    return {
        **row,
        "job_status": "blocked_route_prior",
        "execution_ready": False,
        "route_prior_blocked": True,
        "skip_reason": "route_prior_blocks_repeated_bad_candidate_route",
        "route_prior": {
            "stem_type": str(prior.get("stem_type") or ""),
            "route_id": str(prior.get("route_id") or ""),
            "recommendation": str(prior.get("recommendation") or ""),
            "requires_new_source_candidate": bool(prior.get("requires_new_source_candidate")),
            "candidate_count": int(prior.get("candidate_count") or 0),
            "blocked_count": int(prior.get("blocked_count") or 0),
            "accepted_count": int(prior.get("accepted_count") or 0),
            "top_blockers": dict(prior.get("top_blockers") or {}),
            "top_failure_reasons": dict(prior.get("top_failure_reasons") or {}),
            "source_pool_eligible": bool(prior.get("source_pool_eligible")),
            "source_pool_status": str(prior.get("source_pool_status") or ""),
            "source_pool_reasons": list(prior.get("source_pool_reasons") or []),
            "best_source_track_id": str(prior.get("best_source_track_id") or ""),
            "best_source_candidate_id": str(prior.get("best_source_candidate_id") or ""),
            "best_source_mix_path": str(prior.get("best_source_mix_path") or ""),
            "best_source_vocal_path": str(prior.get("best_source_vocal_path") or ""),
            "best_source_audio_path": str(prior.get("best_source_audio_path") or ""),
            "best_source_score": float(prior.get("best_source_score") or 0.0),
            "best_source_vocal_leakage_in_minus_score": float(
                prior.get("best_source_vocal_leakage_in_minus_score") or 0.0
            ),
            "best_source_mixture_consistency_score": float(prior.get("best_source_mixture_consistency_score") or 0.0),
        },
    }


def direct_minus_job(
    *,
    corpus_row: dict[str, Any],
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    track_id = str(corpus_row.get("track_id") or corpus_row.get("relative_path") or "track")
    audio_path = source_audio_path(corpus_row)
    output_dir = output_root / safe_slug(track_id) / "minus" / "export_instrumental_resurrection_minus"
    listen_dir = output_dir / "listen_diagnostic"
    availability = minus_baseline_availability()
    source_exists = path_exists(audio_path)
    status = job_status_for(source_exists=source_exists, availability_status=str(availability["status"]))
    command = [
        "python",
        "/app/tools/export_instrumental_resurrection_minus.py",
        audio_path,
        str(output_dir),
        "--device",
        str(device),
        "--listen-dir",
        str(listen_dir),
    ]
    return base_job_row(
        corpus_row=corpus_row,
        target_stem="minus",
        route_name="export_instrumental_resurrection_minus",
        route_family="direct_clean_minus_baseline",
        model_id="instrumental_resurrection_minus",
        model_display_name="Instrumental Resurrection minus",
        output_dir=output_dir,
        listen_dir=listen_dir,
        command=command,
        expected_candidate_manifest=output_dir / "stem_manifest.json",
        execution_manifest_path=output_dir / "stem_manifest.json",
        availability=availability,
        job_status=status,
        source_exists=source_exists,
        route_contract="documented_clean_minus_baseline",
    )


def source_pool_repair_row(*, prior: dict[str, Any], output_root: Path) -> dict[str, Any]:
    route_id = str(prior.get("route_id") or "")
    track_id = str(prior.get("best_source_track_id") or "")
    candidate_id = str(prior.get("best_source_candidate_id") or "")
    mix_path = str(prior.get("best_source_mix_path") or "")
    vocal_path = str(prior.get("best_source_vocal_path") or "")
    minus_path = str(prior.get("best_source_audio_path") or "")
    output_dir = output_root / safe_slug(track_id or "source_pool") / "minus_source_pool_phase_repair" / safe_slug(route_id)
    source_paths_exist = all(path_exists(path) for path in (mix_path, vocal_path, minus_path))
    job_status = "queued" if source_paths_exist else "blocked_missing_source_audio"
    return {
        "schema": "testing_stuff_music_corpus_job_row.v1",
        "track_index": -1,
        "track_id": track_id,
        "relative_path": "",
        "source_audio_path": mix_path,
        "host_audio_path_hint": "",
        "source_exists": source_paths_exist,
        "target_stem": "minus",
        "phase": "source_pool_phase_repair_candidate",
        "route_name": route_id,
        "route_family": "cpu_dsp_minus_phase_repair",
        "route_contract": "no_reference_minus_phase_repair_candidate",
        "model_id": route_id,
        "model_display_name": route_id,
        "job_status": job_status,
        "execution_ready": job_status == "queued",
        "planned_only": True,
        "execution_started": False,
        "expected_acceleration_mode": CPU_DSP_ACCELERATION_MODE,
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "cpu_only_dsp_route",
        "output_dir": str(output_dir),
        "listen_dir": str(output_dir / "listen_diagnostic"),
        "command": [
            "python",
            "/app/tools/run_no_reference_minus_phase_repair.py",
            "--mix-path",
            mix_path,
            "--vocal-path",
            vocal_path,
            "--minus-path",
            minus_path,
            "--output-dir",
            str(output_dir),
            "--track-id",
            track_id,
            "--candidate-id",
            candidate_id,
            "--minus-route-id",
            route_id,
        ],
        "expected_candidate_manifest": str(output_dir / "minus_phase_repair_manifest.json"),
        "execution_manifest_path": str(output_dir / "minus_phase_repair_manifest.json"),
        "availability": {
            "status": "available" if source_paths_exist else "unavailable",
            "missing_paths": [path for path in (mix_path, vocal_path, minus_path) if not path_exists(path)],
            "missing_python_modules": [],
            "runtime_blockers": [],
        },
        "source_pool": {
            "stem_type": "minus",
            "route_id": route_id,
            "source_pool_status": str(prior.get("source_pool_status") or ""),
            "source_pool_reasons": list(prior.get("source_pool_reasons") or []),
            "best_source_track_id": track_id,
            "best_source_candidate_id": candidate_id,
            "best_source_mix_path": mix_path,
            "best_source_vocal_path": vocal_path,
            "best_source_audio_path": str(prior.get("best_source_audio_path") or ""),
            "best_source_score": float(prior.get("best_source_score") or 0.0),
            "best_source_vocal_leakage_in_minus_score": float(
                prior.get("best_source_vocal_leakage_in_minus_score") or 0.0
            ),
            "best_source_mixture_consistency_score": float(prior.get("best_source_mixture_consistency_score") or 0.0),
            "source_contract": "repair_source_only_not_delivery_candidate",
        },
        "required_downstream_gates": [
            "source_audio_path_revalidation",
            "minus_phase_repair_variant_generation",
            "no_reference_windowed_judge",
            "artifact_status_contract",
        ],
        "training_allowed": False,
        "fine_tune_allowed": False,
        "can_start_training_now": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "delivery_allowed": False,
        "release_ready": False,
    }


def registry_job(
    *,
    corpus_row: dict[str, Any],
    target_stem: str,
    suggestion: dict[str, Any],
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    track_id = str(corpus_row.get("track_id") or corpus_row.get("relative_path") or "track")
    audio_path = source_audio_path(corpus_row)
    model_id = str(suggestion["model_id"])
    output_dir = output_root / safe_slug(track_id) / target_stem / safe_slug(model_id)
    listen_dir = output_dir / "listen_diagnostic"
    availability = suggestion.get("availability") if isinstance(suggestion.get("availability"), dict) else {}
    source_exists = path_exists(audio_path)
    status = job_status_for(source_exists=source_exists, availability_status=str(availability.get("status") or "unavailable"))
    command = [
        "python",
        "/app/tools/run_registered_model_candidate.py",
        audio_path,
        str(output_dir),
        "--model-id",
        model_id,
        "--device",
        str(device),
        "--listen-dir",
        str(listen_dir),
        "--target-stem",
        target_stem,
    ]
    return base_job_row(
        corpus_row=corpus_row,
        target_stem=target_stem,
        route_name=model_id,
        route_family=str(suggestion.get("family") or "registered_model"),
        model_id=model_id,
        model_display_name=str(suggestion.get("display_name") or model_id),
        output_dir=output_dir,
        listen_dir=listen_dir,
        command=command,
        expected_candidate_manifest=output_dir / "stem_manifest.json",
        execution_manifest_path=output_dir / "registry_candidate_execution_manifest.json",
        availability=availability,
        job_status=status,
        source_exists=source_exists,
        route_contract="registered_model_candidate_manifest_required",
    )


def base_job_row(
    *,
    corpus_row: dict[str, Any],
    target_stem: str,
    route_name: str,
    route_family: str,
    model_id: str,
    model_display_name: str,
    output_dir: Path,
    listen_dir: Path,
    command: list[str],
    expected_candidate_manifest: Path,
    execution_manifest_path: Path,
    availability: dict[str, Any],
    job_status: str,
    source_exists: bool,
    route_contract: str,
) -> dict[str, Any]:
    track_id = str(corpus_row.get("track_id") or corpus_row.get("relative_path") or "track")
    acceleration_contract = acceleration_contract_for_route_family(route_family)
    return {
        "schema": "testing_stuff_music_corpus_job_row.v1",
        "track_index": int(corpus_row.get("index") or 0),
        "track_id": track_id,
        "relative_path": str(corpus_row.get("relative_path") or ""),
        "source_audio_path": source_audio_path(corpus_row),
        "host_audio_path_hint": str(corpus_row.get("host_audio_path_hint") or ""),
        "source_exists": bool(source_exists),
        "target_stem": target_stem,
        "phase": "candidate_generation",
        "route_name": route_name,
        "route_family": route_family,
        "route_contract": route_contract,
        "model_id": model_id,
        "model_display_name": model_display_name,
        "job_status": job_status,
        "execution_ready": job_status == "queued",
        "planned_only": True,
        "execution_started": False,
        "expected_acceleration_mode": acceleration_contract["expected_acceleration_mode"],
        "gpu_inference_required": acceleration_contract["gpu_inference_required"],
        "gpu_inference_skipped": acceleration_contract["gpu_inference_skipped"],
        "gpu_inference_skip_reason": acceleration_contract["gpu_inference_skip_reason"],
        "output_dir": str(output_dir),
        "listen_dir": str(listen_dir),
        "command": command,
        "expected_candidate_manifest": str(expected_candidate_manifest),
        "execution_manifest_path": str(execution_manifest_path),
        "availability": {
            "status": str(availability.get("status") or "unavailable"),
            "missing_paths": list(availability.get("missing_paths") or []),
            "missing_python_modules": list(availability.get("missing_python_modules") or []),
            "runtime_blockers": list(availability.get("runtime_blockers") or []),
        },
        "required_downstream_gates": downstream_gates(target_stem),
        "training_allowed": False,
        "fine_tune_allowed": False,
        "can_start_training_now": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "delivery_allowed": False,
        "release_ready": False,
    }


def registry_suggestions(
    *,
    registry: ModelRegistry,
    target_stem: str,
    max_models_per_stem: int,
    evaluated_labels: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    evaluated = {str(label).lower() for label in evaluated_labels or []}
    candidate_suggestions = registry.suggest_next_candidates(
        stem_names=[registry_stem_name(target_stem)],
        failed_labels=[],
        evaluated_labels=[],
        include_unavailable=True,
        limit=max(0, len(registry.entries)),
    )
    suggestion_dicts = [
        suggestion.to_dict()
        for suggestion in candidate_suggestions
        if not (evaluated & entry_known_labels(registry.get(str(suggestion.model_id))))
        and not (target_stem == "minus" and entry_is_stem_lane_minus_route(registry.get(str(suggestion.model_id))))
    ]
    suggestion_dicts.sort(key=suggestion_dict_sort_key)
    return suggestion_dicts[: max(0, int(max_models_per_stem))]


def entry_is_stem_lane_minus_route(entry: ModelRegistryEntry | None) -> bool:
    if entry is None:
        return False
    command_text = " ".join(str(part) for part in entry.command_template).lower()
    return any(exporter in command_text for exporter in STEM_LANE_MINUS_EXPORTERS)


def jobs_for_row(
    *,
    corpus_row: dict[str, Any],
    output_root: Path,
    registry: ModelRegistry,
    target_stems: Sequence[str] | None,
    max_models_per_stem: int,
    device: str,
    route_priors: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    route_priors = route_priors or {}
    rows: list[dict[str, Any]] = []
    for stem in row_stems(corpus_row, target_stems):
        if stem == "minus":
            direct_row = direct_minus_job(corpus_row=corpus_row, output_root=output_root, device=device)
            rows.append(direct_row)
            if matching_blocking_route_prior(direct_row, route_priors):
                blocked_route_ids = sorted(blocked_route_ids_for_stem(route_priors, "minus"))
                for suggestion in registry_suggestions(
                    registry=registry,
                    target_stem=stem,
                    max_models_per_stem=max_models_per_stem,
                    evaluated_labels=blocked_route_ids,
                ):
                    rows.append(
                        registry_job(
                            corpus_row=corpus_row,
                            target_stem=stem,
                            suggestion=suggestion,
                            output_root=output_root,
                            device=device,
                        )
                    )
            continue
        for suggestion in registry_suggestions(
            registry=registry,
            target_stem=stem,
            max_models_per_stem=max_models_per_stem,
        ):
            rows.append(
                registry_job(
                    corpus_row=corpus_row,
                    target_stem=stem,
                    suggestion=suggestion,
                    output_root=output_root,
                    device=device,
                )
            )
    return rows


def build_summary(rows: list[dict[str, Any]], target_stems: Sequence[str]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("job_status") or "unknown") for row in rows)
    track_ids = sorted({str(row.get("track_id") or "") for row in rows if row.get("track_id")})
    acceleration_modes = sorted({str(row.get("expected_acceleration_mode") or "") for row in rows if row.get("expected_acceleration_mode")})
    return {
        "schema": "testing_stuff_music_corpus_job_matrix_summary.v1",
        "total_tracks": len(track_ids),
        "total_jobs": len(rows),
        "target_stems": list(target_stems),
        "queued": int(status_counts.get("queued", 0)),
        "blocked_model_unavailable": int(status_counts.get("blocked_model_unavailable", 0)),
        "blocked_missing_source_audio": int(status_counts.get("blocked_missing_source_audio", 0)),
        "blocked_route_prior": int(status_counts.get("blocked_route_prior", 0)),
        "source_pool_repair_candidates": sum(
            1 for row in rows if row.get("phase") == "source_pool_phase_repair_candidate"
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "planned_model_ids": sorted({str(row.get("model_id") or "") for row in rows if row.get("model_id")}),
        "execution_started": False,
        "expected_acceleration_mode": acceleration_modes[0] if len(acceleration_modes) == 1 else "mixed",
        "expected_acceleration_modes": acceleration_modes,
        "safety": dict(SAFETY_FLAGS),
    }


def build_manifest(output_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "testing_stuff_music_corpus_job_matrix_manifest.v1",
        "manifest_type": "testing_stuff_music_corpus_job_matrix",
        "output_root": str(output_root),
        "total_jobs": len(rows),
        "job_matrix_status": "ready" if rows else "blocked_no_jobs",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_job_matrix_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "cleanup_routes_added": False,
        "thresholds_relaxed": False,
        "audio_copied": False,
        "audio_exported": False,
        "audio_hashed": False,
        "execution_started": False,
        "summary": summary,
        "safety": dict(SAFETY_FLAGS),
        "output_files": {
            "jobs_jsonl": str(output_root / "testing_stuff_music_corpus_job_matrix.jsonl"),
            "manifest_json": str(output_root / "testing_stuff_music_corpus_job_matrix_manifest.json"),
            "summary_md": str(output_root / "testing_stuff_music_corpus_job_matrix_summary.md"),
            "progress_jsonl": str(output_root / "testing_stuff_music_corpus_job_matrix_progress.jsonl"),
        },
    }


def build_job_matrix(
    *,
    corpus_plan_jsonl: Path,
    output_root: Path,
    registry: ModelRegistry | None = None,
    target_stems: Sequence[str] | None = None,
    max_models_per_stem: int = DEFAULT_MAX_MODELS_PER_STEM,
    device: str = "cuda",
    route_priors_jsonl: Path | None = None,
) -> dict[str, Any]:
    registry = registry or default_model_registry()
    plan_rows = list(iter_jsonl(corpus_plan_jsonl))
    route_priors = load_route_prior_index(route_priors_jsonl)
    selected_stems = [stem for stem in DEFAULT_TARGET_STEMS if stem in set(target_stems or DEFAULT_TARGET_STEMS)]
    rows: list[dict[str, Any]] = []
    for corpus_row in sorted(plan_rows, key=lambda row: (int(row.get("index") or 0), str(row.get("track_id") or ""))):
        row_jobs = jobs_for_row(
            corpus_row=corpus_row,
            output_root=output_root,
            registry=registry,
            target_stems=selected_stems,
            max_models_per_stem=max_models_per_stem,
            device=device,
            route_priors=route_priors,
        )
        rows.extend(apply_route_prior_policy(row, route_priors) for row in row_jobs)
    if "minus" in selected_stems:
        rows.extend(
            source_pool_repair_row(prior=prior, output_root=output_root)
            for prior in source_pool_route_priors(route_priors, "minus")
        )
    rows.sort(key=lambda row: (row["track_index"], row["track_id"], row["target_stem"], row["model_id"]))
    summary = build_summary(rows, selected_stems)
    manifest = build_manifest(output_root, rows, summary)
    return {"rows": rows, "summary": summary, "manifest": manifest}


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Testing Stuff Music Corpus Job Matrix",
        "",
        f"- total_tracks: {summary['total_tracks']}",
        f"- total_jobs: {summary['total_jobs']}",
        f"- target_stems: {', '.join(summary['target_stems'])}",
        f"- queued: {summary['queued']}",
        f"- blocked_model_unavailable: {summary['blocked_model_unavailable']}",
        f"- blocked_missing_source_audio: {summary['blocked_missing_source_audio']}",
        f"- blocked_route_prior: {summary['blocked_route_prior']}",
        f"- source_pool_repair_candidates: {summary['source_pool_repair_candidates']}",
        f"- execution_started: {str(summary['execution_started']).lower()}",
        f"- expected_acceleration_mode: {summary['expected_acceleration_mode']}",
        "",
        "## Safety",
        "",
    ]
    for key, value in summary["safety"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def progress_event(stage: str, *, started_at: float, status: str = "progress", **metadata: Any) -> dict[str, Any]:
    return {
        "type": "testing_stuff_music_corpus_job_matrix_progress",
        "stage": stage,
        "status": status,
        "since_start_sec": round(max(0.0, time.perf_counter() - started_at), 6),
        "metadata": metadata,
    }


def write_outputs(result: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    progress_path = output_root / "testing_stuff_music_corpus_job_matrix_progress.jsonl"
    progress_rows = [
        progress_event(
            "start",
            started_at=started_at,
            total_jobs=len(result["rows"]),
            expected_acceleration_mode=result["summary"].get("expected_acceleration_mode"),
            expected_acceleration_modes=result["summary"].get("expected_acceleration_modes"),
            gpu_inference_required=any(row.get("gpu_inference_required") is True for row in result["rows"]),
        ),
        progress_event(
            "write_outputs",
            started_at=started_at,
            status="completed",
            output_root=str(output_root),
        ),
    ]
    write_jsonl(output_root / "testing_stuff_music_corpus_job_matrix.jsonl", result["rows"])
    write_json(output_root / "testing_stuff_music_corpus_job_matrix_manifest.json", result["manifest"])
    write_summary_md(output_root / "testing_stuff_music_corpus_job_matrix_summary.md", result["summary"])
    write_jsonl(progress_path, progress_rows)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_testing_stuff_music_corpus_job_matrix.py",
        entrypoint_path="/app/tools/build_testing_stuff_music_corpus_job_matrix.py",
    )
    parser = argparse.ArgumentParser(description="Build planned no-reference candidate-generation jobs from testing_stuff_music corpus plan.")
    parser.add_argument("--corpus-plan-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=Path("/app"))
    parser.add_argument("--stem", action="append", default=[])
    parser.add_argument("--max-models-per-stem", type=int, default=DEFAULT_MAX_MODELS_PER_STEM)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--route-priors-jsonl", type=Path, default=None)
    args = parser.parse_args(argv)

    result = build_job_matrix(
        corpus_plan_jsonl=args.corpus_plan_jsonl,
        output_root=args.output_root,
        registry=default_model_registry(runtime_root=args.runtime_root),
        target_stems=args.stem or None,
        max_models_per_stem=args.max_models_per_stem,
        device=str(args.device),
        route_priors_jsonl=args.route_priors_jsonl,
    )
    write_outputs(result, args.output_root)
    print(stable_json(result["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
