from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_ROOT.parent
SCRIPTS_ROOT = ROOT_DIR / "scripts"
for search_path in (ROOT_DIR, SCRIPT_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from docker_runtime import require_docker_runtime  # noqa: E402
import route_drum_model_zoo_segments as segment_router  # noqa: E402
import run_ground_truth_quality_gate as quality_gate  # noqa: E402
import run_model_zoo_benchmark as benchmark  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_candidate(value: str) -> segment_router.CandidateSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must use LABEL=PATH format")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("candidate label cannot be empty")
    return segment_router.CandidateSpec(label=label, stem_path=Path(raw_path))


def _load_profile_config(profile_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = _load_json(profile_path)
    contract = profile.get("contract", {})
    if contract.get("target_reference_assisted") is not True:
        raise ValueError("holdout profile must be target-reference-assisted calibration output.")
    if contract.get("calibration_only") is not True:
        raise ValueError("holdout profile must be marked calibration_only.")
    if contract.get("same_track_delivery_allowed") is not False:
        raise ValueError("holdout profile must forbid same-track delivery.")
    if contract.get("application_uses_target_reference") is not False:
        raise ValueError("holdout profile must not use target reference during application.")
    recommended_config = profile.get("recommended_router_config")
    if not isinstance(recommended_config, dict):
        raise ValueError("holdout profile missing recommended_router_config.")
    return profile, recommended_config


def _router_config_from_profile(
    *,
    profile_path: Path,
    recommended_config: dict[str, Any],
    base_config: segment_router.RouterConfig | None,
) -> segment_router.RouterConfig:
    config = base_config or segment_router.RouterConfig()
    return replace(
        config,
        baseline_candidate_label=str(recommended_config["baseline_candidate_label"]),
        baseline_switch_margin=float(recommended_config["baseline_switch_margin"]),
        baseline_max_switch_fraction=(
            None
            if recommended_config.get("baseline_max_switch_fraction") is None
            else float(recommended_config["baseline_max_switch_fraction"])
        ),
        baseline_calibration_profile_path=str(profile_path),
    )


def _failed_check_names(benchmark_manifest: dict[str, Any]) -> list[str]:
    if not benchmark_manifest.get("ranked_candidates"):
        return []
    return list(benchmark_manifest["ranked_candidates"][0].get("failed_check_names", []))


def _stem_score(benchmark_manifest: dict[str, Any]) -> dict[str, Any]:
    if not benchmark_manifest.get("ranked_candidates"):
        return {}
    stem_scores = benchmark_manifest["ranked_candidates"][0].get("stem_scores", [])
    if not stem_scores:
        return {}
    return dict(stem_scores[0])


def _validation_status(benchmark_manifest: dict[str, Any]) -> str:
    if benchmark_manifest.get("selected_candidate") is not None:
        return "validated"
    return "rejected_for_generalization"


def validate_holdout_profile(
    *,
    profile_path: Path,
    ground_truth_manifest_path: Path,
    candidates: Sequence[segment_router.CandidateSpec],
    output_dir: Path,
    stem_name: str,
    router_config: segment_router.RouterConfig | None = None,
    benchmark_thresholds: quality_gate.ObjectiveThresholds | None = None,
    sample_rate: int = quality_gate.DEFAULT_SAMPLE_RATE,
    max_offset_seconds: float = 0.0,
    output_label: str | None = None,
    next_candidate_limit: int = 0,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one holdout candidate is required")
    profile, recommended_config = _load_profile_config(profile_path)
    profile_stem_name = str(profile.get("stem_name", "")).strip()
    if profile_stem_name and profile_stem_name != stem_name:
        raise ValueError(f"profile stem_name does not match holdout stem: {profile_stem_name} != {stem_name}")
    applied_router_config = _router_config_from_profile(
        profile_path=profile_path,
        recommended_config=recommended_config,
        base_config=router_config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    route_output_dir = output_dir / "applied_route"
    benchmark_output_dir = output_dir / "benchmark"
    route_manifest = segment_router.route_drum_segments(
        list(candidates),
        route_output_dir,
        output_label=output_label or f"holdout_profile_applied_{stem_name}.flac",
        stem_name=stem_name,
        target_reference_path=None,
        config=applied_router_config,
    )
    candidate_manifest_path = route_output_dir / "candidate_manifest.json"
    benchmark_manifest = benchmark.run_model_zoo_benchmark(
        ground_truth_manifest=ground_truth_manifest_path,
        candidate_manifests=[
            benchmark.CandidateManifestSpec(
                label=f"holdout_profile_applied_{stem_name}",
                manifest_path=candidate_manifest_path,
            )
        ],
        output_dir=benchmark_output_dir,
        stem_names=[stem_name],
        sample_rate=sample_rate,
        max_offset_seconds=max_offset_seconds,
        thresholds=benchmark_thresholds or quality_gate.ObjectiveThresholds(),
        include_unavailable_registry_candidates=False,
        next_candidate_limit=next_candidate_limit,
    )
    status = _validation_status(benchmark_manifest)
    stem_score = _stem_score(benchmark_manifest)
    failed_check_names = _failed_check_names(benchmark_manifest)
    validation = {
        "status": status,
        "reason": (
            "applied reference-free profile passed the verified holdout objective gate"
            if status == "validated"
            else "applied reference-free profile failed the verified holdout objective gate"
        ),
        "failed_check_names": failed_check_names,
        "reference_likeness_percent": stem_score.get("reference_likeness_percent"),
        "sdr_db": stem_score.get("sdr_db"),
        "si_sdr_db": stem_score.get("si_sdr_db"),
        "sar_db": stem_score.get("sar_db"),
    }
    manifest = {
        "schema_version": 1,
        "contract": {
            "route": "segment_router_confidence_holdout_validation_v1",
            "profile_path": str(profile_path),
            "profile_type": profile.get("profile_type"),
            "stem_name": stem_name,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "reference_assisted_evaluation": True,
            "same_track_delivery_allowed": False,
            "requires_distinct_holdout_track": True,
            "fallback_used": False,
            "quality_gate_required": True,
            "delivery_allowed_without_holdout_validation": False,
        },
        "inputs": {
            "ground_truth_manifest_path": str(ground_truth_manifest_path),
            "candidate_labels": [candidate.label for candidate in candidates],
            "sample_rate": sample_rate,
            "max_offset_seconds": max_offset_seconds,
            "next_candidate_limit": next_candidate_limit,
        },
        "recommended_router_config": recommended_config,
        "applied_route": {
            "manifest_path": str(route_output_dir / "drum_model_zoo_segment_router_manifest.json"),
            "candidate_manifest_path": str(candidate_manifest_path),
            "contract": route_manifest["contract"],
            "output": route_manifest["output"],
        },
        "benchmark": {
            "manifest_path": str(benchmark_output_dir / "model_zoo_benchmark_manifest.json"),
            "overall": benchmark_manifest["overall"],
            "selected_candidate": benchmark_manifest["selected_candidate"],
            "ranked_candidates": benchmark_manifest["ranked_candidates"],
        },
        "validation": validation,
    }
    manifest_path = output_dir / "holdout_validation_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a segment-router calibration profile on a holdout track.")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=_parse_candidate, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--output-label")
    parser.add_argument("--sample-rate", type=int, default=quality_gate.DEFAULT_SAMPLE_RATE)
    parser.add_argument("--max-offset-seconds", type=float, default=0.0)
    parser.add_argument("--segment-seconds", type=float, default=segment_router.RouterConfig.segment_seconds)
    parser.add_argument("--crossfade-seconds", type=float, default=segment_router.RouterConfig.crossfade_seconds)
    parser.add_argument("--min-rms-dbfs", type=float, default=segment_router.RouterConfig.min_rms_dbfs)
    parser.add_argument("--switch-hysteresis", type=float, default=segment_router.RouterConfig.switch_hysteresis)
    parser.add_argument("--min-sdr-db", type=float, default=quality_gate.ObjectiveThresholds.min_sdr_db)
    parser.add_argument("--min-si-sdr-db", type=float, default=quality_gate.ObjectiveThresholds.min_si_sdr_db)
    parser.add_argument("--min-sir-db", type=float, default=quality_gate.ObjectiveThresholds.min_sir_db)
    parser.add_argument("--min-sar-db", type=float, default=quality_gate.ObjectiveThresholds.min_sar_db)
    parser.add_argument(
        "--min-reference-likeness-percent",
        type=float,
        default=quality_gate.ObjectiveThresholds.min_reference_likeness_percent,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="validate_segment_router_confidence_holdout.py",
        entrypoint_path="/app/tools/validate_segment_router_confidence_holdout.py",
    )
    args = _build_arg_parser().parse_args(argv)
    manifest = validate_holdout_profile(
        profile_path=args.profile,
        ground_truth_manifest_path=args.ground_truth_manifest,
        candidates=args.candidate,
        output_dir=args.output_dir,
        stem_name=str(args.stem),
        router_config=segment_router.RouterConfig(
            segment_seconds=float(args.segment_seconds),
            crossfade_seconds=float(args.crossfade_seconds),
            min_rms_dbfs=float(args.min_rms_dbfs),
            switch_hysteresis=float(args.switch_hysteresis),
        ),
        benchmark_thresholds=quality_gate.ObjectiveThresholds(
            min_sdr_db=float(args.min_sdr_db),
            min_si_sdr_db=float(args.min_si_sdr_db),
            min_sir_db=float(args.min_sir_db),
            min_sar_db=float(args.min_sar_db),
            min_reference_likeness_percent=float(args.min_reference_likeness_percent),
        ),
        sample_rate=int(args.sample_rate),
        max_offset_seconds=float(args.max_offset_seconds),
        output_label=args.output_label,
    )
    print(
        json.dumps(
            {
                "manifest_path": manifest["manifest_path"],
                "validation": manifest["validation"],
                "benchmark_manifest": manifest["benchmark"]["manifest_path"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
