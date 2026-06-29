"""Apply a teacher-mask checkpoint as a stem-search candidate route."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.docker_runtime import require_docker_runtime
from tools.train_teacher_mask_separator import (
    apply_teacher_mask_checkpoint_to_source,
    audio_summary,
    sha256_file,
    write_json,
)


def apply_teacher_mask_checkpoint_candidate(
    *,
    source_mix_path: Path,
    stems_dir: Path,
    checkpoint_path: Path,
    stem: str,
    device: str = "cuda",
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    stems_dir.mkdir(parents=True, exist_ok=True)
    apply_dir = stems_dir / "tm"
    progress_path = stems_dir / "tm_progress.jsonl"
    manifest = apply_teacher_mask_checkpoint_to_source(
        source_mix_path=source_mix_path,
        checkpoint_path=checkpoint_path,
        output_dir=apply_dir,
        stem_name=stem,
        device=device,
        command_args=command_args or [],
        progress_path=progress_path,
        progress_context={"route_scope": "stem_search_teacher_mask_checkpoint_candidate"},
    )
    source_stem_path = Path(str(manifest["stems"][stem]["path"]))
    output_path = stems_dir / f"{stem}.flac"
    shutil.copyfile(source_stem_path, output_path)
    result = {
        "schema": "teacher_mask_checkpoint_candidate.v1",
        "source_mix": audio_summary(source_mix_path),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "bytes": int(checkpoint_path.stat().st_size),
        },
        "stem": stem,
        "stem_path": str(output_path),
        "stem_audio": audio_summary(output_path),
        "application_manifest_path": str(apply_dir / "stem_manifest.json"),
        "progress_jsonl": str(progress_path),
        "contract": {
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "quality_gate_required": True,
            "delivery_allowed_without_runner_gates": False,
            "candidate_route_only": True,
        },
        "delivery_allowed": False,
        "presentation_status": "candidate_route_not_listen_pack",
    }
    write_json(stems_dir / "tm_candidate.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_mix", type=Path)
    parser.add_argument("stems_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="apply_teacher_mask_checkpoint_candidate.py",
        entrypoint_path="/app/tools/apply_teacher_mask_checkpoint_candidate.py",
    )
    args = parse_args() if argv is None else parse_args_from(argv)
    command_args = list(sys.argv[1:] if argv is None else argv)
    result = apply_teacher_mask_checkpoint_candidate(
        source_mix_path=args.source_mix,
        stems_dir=args.stems_dir,
        checkpoint_path=args.checkpoint,
        stem=str(args.stem),
        device=str(args.device),
        command_args=command_args,
    )
    print(json.dumps({"manifest_path": str(Path(args.stems_dir) / "tm_candidate.json"), "stem_path": result["stem_path"]}))
    return 0


def parse_args_from(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_mix", type=Path)
    parser.add_argument("stems_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
