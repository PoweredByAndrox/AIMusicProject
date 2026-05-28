from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from worker.model_registry import default_model_registry  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List registered source-separation model candidates.")
    parser.add_argument("--runtime-root", type=Path, default=Path("/app"))
    parser.add_argument("--stem", action="append")
    parser.add_argument("--failed-label", action="append", default=[])
    parser.add_argument("--evaluated-label", action="append", default=[])
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="list_model_registry.py",
        entrypoint_path="/app/tools/list_model_registry.py",
    )
    args = build_arg_parser().parse_args(argv)
    if args.limit < 0:
        raise ValueError("--limit must not be negative")

    registry = default_model_registry(runtime_root=args.runtime_root)
    if args.stem:
        suggestions = registry.suggest_next_candidates(
            stem_names=args.stem,
            failed_labels=args.failed_label,
            evaluated_labels=args.evaluated_label,
            include_unavailable=not bool(args.available_only),
            limit=args.limit,
        )
        payload = {
            "schema_version": 1,
            "mode": "suggestions",
            "runtime_root": str(args.runtime_root),
            "requested_stems": args.stem,
            "failed_labels": args.failed_label,
            "evaluated_labels": args.evaluated_label,
            "suggestions": [suggestion.to_dict() for suggestion in suggestions],
        }
    else:
        payload = registry.to_manifest()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
