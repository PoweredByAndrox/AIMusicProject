"""Run a Docker-only stem contamination learning/search cycle."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_stem_contamination_dataset import build_dataset_from_manifest
from tools.docker_runtime import require_docker_runtime
from tools.run_fully_automatic_stem_search import (
    CommandRunner,
    build_default_from_scratch_recipe,
    run_fully_automatic_stem_search,
)
from tools.stem_contamination_common import validate_no_artifact_rag_output
from tools.train_stem_contamination_detector import train_stem_contamination_detector


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _recipe_with_detector_manifest(
    *,
    recipe_path: Path | None,
    detector_manifest_path: Path,
    output_path: Path,
    demucs_available: bool | None,
) -> Path:
    if recipe_path is not None:
        recipe = _read_json(recipe_path)
    else:
        recipe = build_default_from_scratch_recipe(demucs_available=demucs_available)
    recipe["contamination_model_manifest"] = str(detector_manifest_path)
    _write_json(output_path, recipe)
    return output_path


def run_stem_contamination_learning_cycle(
    *,
    mix_path: Path,
    clean_pairs_manifest: Path,
    output_dir: Path,
    recipe_path: Path | None = None,
    command_runner: CommandRunner | None = None,
    contamination_db: float = -35.0,
    training_device: str = "cuda",
    training_epochs: int = 120,
    training_hidden_size: int = 8,
    training_min_positive_rows: int = 2,
    max_candidates: int = 10,
    require_gpu: bool = True,
    no_artifact_rag: bool = True,
    demucs_available: bool | None = None,
) -> dict[str, Any]:
    if no_artifact_rag:
        validate_no_artifact_rag_output(output_dir)
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_jsonl(
        progress_path,
        {
            "event": "start",
            "mix_path": str(mix_path),
            "clean_pairs_manifest": str(clean_pairs_manifest),
            "artifact_rag_write_allowed": not no_artifact_rag,
            "gpu_required": require_gpu,
        },
    )

    dataset_dir = output_dir / "dataset"
    model_dir = output_dir / "model"
    search_dir = output_dir / "search"
    rows_jsonl = build_dataset_from_manifest(clean_pairs_manifest, dataset_dir, contamination_db)
    _append_jsonl(progress_path, {"event": "dataset_built", "rows_jsonl": str(rows_jsonl)})
    model_manifest = train_stem_contamination_detector(
        rows_jsonl=rows_jsonl,
        output_dir=model_dir,
        device=training_device,
        epochs=training_epochs,
        hidden_size=training_hidden_size,
        min_positive_rows=training_min_positive_rows,
    )
    model_manifest_path = model_dir / "manifest.json"
    _append_jsonl(
        progress_path,
        {
            "event": "model_trained",
            "model_manifest_json": str(model_manifest_path),
            "training_status": model_manifest.get("training_status"),
            "pair_count": model_manifest.get("pair_count"),
        },
    )
    enriched_recipe_path = _recipe_with_detector_manifest(
        recipe_path=recipe_path,
        detector_manifest_path=model_manifest_path,
        output_path=output_dir / "recipe.json",
        demucs_available=demucs_available,
    )
    search_manifest = run_fully_automatic_stem_search(
        mix_path=mix_path,
        output_dir=search_dir,
        max_candidates=max_candidates,
        require_gpu=require_gpu,
        no_artifact_rag=no_artifact_rag,
        recipe_path=enriched_recipe_path,
        command_runner=command_runner,
        allow_default_recipe=False,
    )
    search_run_json = search_dir / "run.json"
    cycle_status = str(search_manifest.get("verdict") or "blocked_unknown")
    manifest = {
        "schema": "stem_contamination_learning_cycle.v1",
        "cycle_status": cycle_status,
        "mix_path": str(mix_path),
        "clean_pairs_manifest": str(clean_pairs_manifest),
        "output_dir": str(output_dir),
        "dataset_rows_jsonl": str(rows_jsonl),
        "model_manifest_json": str(model_manifest_path),
        "recipe_json": str(enriched_recipe_path),
        "search_run_json": str(search_run_json),
        "search_output_dir": str(search_dir),
        "search_verdict": cycle_status,
        "model_training_status": str(model_manifest.get("training_status") or ""),
        "model_pair_count": int(model_manifest.get("pair_count") or 0),
        "delivery_status": str(search_manifest.get("delivery_status") or "blocked"),
        "delivery_allowed": bool(search_manifest.get("delivery_allowed")),
        "presentation_status": str(search_manifest.get("presentation_status") or "blocked_from_listen_pack"),
        "artifact_rag_write_allowed": not no_artifact_rag,
        "elapsed_sec": round(time.monotonic() - started, 6),
    }
    _write_json(output_dir / "manifest.json", manifest)
    _append_jsonl(progress_path, {"event": "finish", "cycle_status": cycle_status, "elapsed_sec": manifest["elapsed_sec"]})
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", required=True)
    parser.add_argument("--clean-pairs-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--recipe")
    parser.add_argument("--contamination-db", type=float, default=-35.0)
    parser.add_argument("--training-device", default="cuda")
    parser.add_argument("--training-epochs", type=int, default=120)
    parser.add_argument("--training-hidden-size", type=int, default=8)
    parser.add_argument("--training-min-positive-rows", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--require-gpu", default="true")
    parser.add_argument("--no-artifact-rag", action="store_true")
    return parser.parse_args(argv)


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_stem_contamination_learning_cycle.py",
        entrypoint_path="/app/tools/run_stem_contamination_learning_cycle.py",
    )
    args = parse_args(argv)
    manifest = run_stem_contamination_learning_cycle(
        mix_path=Path(args.mix),
        clean_pairs_manifest=Path(args.clean_pairs_manifest),
        output_dir=Path(args.out),
        recipe_path=Path(args.recipe) if args.recipe else None,
        contamination_db=float(args.contamination_db),
        training_device=str(args.training_device),
        training_epochs=int(args.training_epochs),
        training_hidden_size=int(args.training_hidden_size),
        training_min_positive_rows=int(args.training_min_positive_rows),
        max_candidates=int(args.max_candidates),
        require_gpu=_parse_bool(args.require_gpu),
        no_artifact_rag=bool(args.no_artifact_rag),
    )
    print(json.dumps({"manifest_json": str(Path(args.out) / "manifest.json"), "cycle_status": manifest["cycle_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
