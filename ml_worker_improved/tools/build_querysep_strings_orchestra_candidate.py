from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from worker.high_quality_stem_export import export_track_stems_aggressive_melody_first  # noqa: E402
from worker.querysep_long_tail_build import build_querysep_long_tail_stems  # noqa: E402

MODEL_ID = "querysep_banquet_strings_orchestra_v1"
TARGET_STEM = "strings_orchestra"
DEFAULT_QUERYSEP_WORKER_URL = "http://querysep-worker:8000"
DEFAULT_QUERYSEP_OUTPUT_ROOT: str | None = None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_or_convert_to_flac(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() == ".flac":
        shutil.copyfile(source_path, target_path)
        return
    audio, sample_rate = sf.read(str(source_path), always_2d=True, dtype="float32")
    sf.write(str(target_path), audio, int(sample_rate), subtype="PCM_24")


def _no_reference_contract() -> dict[str, Any]:
    return {
        "full_track": True,
        "benchmark_compatible": True,
        "registered_model_candidate": True,
        "target_reference_assisted": False,
        "target_reference_used": False,
        "target_reference_used_at_application": False,
        "application_reference_free": True,
        "delivery_allowed_without_reference": True,
        "fallback_used": False,
        "hidden_inference": False,
        "per_track_hardcode_allowed": False,
        "not_timestamp_hardcoded": True,
        "quality_gate_required": True,
        "route_name": MODEL_ID,
    }


def _default_path_mappings(raw_path_mappings: Sequence[dict[str, str]] | None) -> list[dict[str, str]]:
    if raw_path_mappings:
        return [dict(mapping) for mapping in raw_path_mappings]
    return [{"local_prefix": "/app/lab_artifacts", "querysep_prefix": "/main_lab_artifacts"}]


def build_querysep_strings_orchestra_candidate(
    *,
    source_audio_path: Path,
    output_dir: Path,
    model_name: str = "htdemucs_6s",
    device: str = "cuda",
    querysep_worker_url: str = DEFAULT_QUERYSEP_WORKER_URL,
    querysep_output_root: str | None = DEFAULT_QUERYSEP_OUTPUT_ROOT,
    path_mappings: Sequence[dict[str, str]] | None = None,
    listen_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_source_audio_path = Path(source_audio_path).resolve()
    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    base_output_dir = resolved_output_dir / "base_stems"
    base_manifest = export_track_stems_aggressive_melody_first(
        resolved_source_audio_path,
        base_output_dir,
        model_name=str(model_name),
        device=str(device),
    )
    base_manifest_path = Path(str(base_manifest.get("manifest_path") or base_output_dir / "stem_manifest.json"))

    querysep_output_dir = resolved_output_dir / "querysep_long_tail"
    querysep_result = build_querysep_long_tail_stems(
        audio_path=resolved_source_audio_path,
        output_dir=querysep_output_dir,
        build_id=f"{MODEL_ID}-{resolved_source_audio_path.stem}",
        target_stems=[TARGET_STEM],
        source_stem_manifest_path=str(base_manifest_path),
        querysep_worker_url=str(querysep_worker_url),
        querysep_output_root=str(querysep_output_root) if querysep_output_root else None,
        path_mappings=_default_path_mappings(path_mappings),
        backend="banquet_byoq",
        dry_run=False,
    )

    selected = querysep_result.get("stems", {}).get(TARGET_STEM, {})
    if querysep_result.get("status") != "completed" or selected.get("status") != "accepted":
        failure_reason = querysep_result.get("failure", {}).get("reason") or selected.get("failure", {}).get("reason")
        raise RuntimeError(
            f"QuerySep long-tail route did not accept {TARGET_STEM}: "
            f"status={querysep_result.get('status')} stem_status={selected.get('status')} "
            f"reason={failure_reason or 'unknown'}"
        )
    selected_output_path = Path(str(selected.get("selected_output_path") or selected.get("local_output_path") or ""))
    if not selected_output_path.exists():
        raise FileNotFoundError(f"selected QuerySep output is missing: {selected_output_path}")

    stem_path = resolved_output_dir / f"{TARGET_STEM}.flac"
    _copy_or_convert_to_flac(selected_output_path, stem_path)

    manifest = {
        "schema_version": 1,
        "model": {"model_id": MODEL_ID, "family": "querysep"},
        "model_id": MODEL_ID,
        "model_name": MODEL_ID,
        "source_audio_path": str(resolved_source_audio_path),
        "sample_rate": selected.get("quality_gate", {}).get("metrics", {}).get("output", {}).get("sample_rate"),
        "contract": _no_reference_contract(),
        "pipeline": {
            "route": MODEL_ID,
            "base_model": str(model_name),
            "base_manifest_path": str(base_manifest_path),
            "querysep_backend": "banquet_byoq",
            "querysep_manifest_path": str(querysep_result.get("manifest_path")),
            "querysep_worker_url": str(querysep_worker_url),
            "querysep_output_root": str(querysep_output_root) if querysep_output_root else None,
            "path_mappings": _default_path_mappings(path_mappings),
        },
        "stems": {
            TARGET_STEM: {
                "path": str(stem_path),
                "role": "no_reference_querysep_long_tail_candidate",
                "selected_output_path": str(selected_output_path),
                "quality_gate": selected.get("quality_gate"),
                "router": selected.get("router"),
            }
        },
        "fallback_used": False,
        "presentation_status": "requires_objective_gate",
    }
    manifest_path = resolved_output_dir / "stem_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)

    if listen_dir is not None:
        resolved_listen_dir = Path(listen_dir).resolve()
        resolved_listen_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stem_path, resolved_listen_dir / "01_querysep_strings_orchestra_candidate.flac")
        (resolved_listen_dir / "00_README.txt").write_text(
            "Diagnostic no-reference QuerySep strings_orchestra candidate. "
            "Do not deliver unless objective/RALPH gate accepts it.\n",
            encoding="utf-8",
        )

    return manifest


def parse_path_mapping(raw_mapping: str) -> dict[str, str]:
    if "=" not in raw_mapping:
        raise argparse.ArgumentTypeError("path mapping must use LOCAL_PREFIX=QUERYSEP_PREFIX")
    local_prefix, querysep_prefix = raw_mapping.split("=", 1)
    if not local_prefix.strip() or not querysep_prefix.strip():
        raise argparse.ArgumentTypeError("path mapping prefixes must not be empty")
    return {"local_prefix": local_prefix.strip(), "querysep_prefix": querysep_prefix.strip()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build no-reference QuerySep strings_orchestra candidate.")
    parser.add_argument("source_audio_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-name", default="htdemucs_6s")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--querysep-worker-url", default=DEFAULT_QUERYSEP_WORKER_URL)
    parser.add_argument("--querysep-output-root", default=DEFAULT_QUERYSEP_OUTPUT_ROOT)
    parser.add_argument("--path-mapping", action="append", type=parse_path_mapping)
    parser.add_argument("--listen-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_querysep_strings_orchestra_candidate.py",
        entrypoint_path="/app/tools/build_querysep_strings_orchestra_candidate.py",
    )
    args = build_arg_parser().parse_args(argv)
    manifest = build_querysep_strings_orchestra_candidate(
        source_audio_path=args.source_audio_path,
        output_dir=args.output_dir,
        model_name=str(args.model_name),
        device=str(args.device),
        querysep_worker_url=str(args.querysep_worker_url),
        querysep_output_root=args.querysep_output_root,
        path_mappings=args.path_mapping,
        listen_dir=args.listen_dir,
    )
    print(json.dumps({"manifest_path": manifest["manifest_path"], "contract": manifest["contract"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
