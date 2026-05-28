from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402
from worker.high_quality_stem_export import repair_exported_stem_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair exported HQ stem directories by adding melody and rebalancing leaked melody out of drums/other.")
    parser.add_argument("stem_dirs", nargs="+", help="One or more existing stem export directories containing stem_manifest.json and FLAC stems.")
    parser.add_argument(
        "--output-suffix",
        default="_melody_repaired",
        help="Suffix for the sibling repaired directory. Ignored when --in-place is used.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the source directory instead of writing to a sibling repaired directory.",
    )
    return parser.parse_args()


def main() -> int:
    require_docker_runtime(
        tool_name="repair_hq_stem_exports.py",
        entrypoint_path="/app/tools/repair_hq_stem_exports.py",
    )
    args = parse_args()
    for raw_dir in args.stem_dirs:
        source_dir = Path(raw_dir).resolve()
        target_dir = source_dir if args.in_place else source_dir.with_name(f"{source_dir.name}{args.output_suffix}")
        manifest = repair_exported_stem_directory(source_dir, target_dir)
        print(f"repaired_stem_dir={target_dir}")
        print(f"manifest_path={manifest['manifest_path']}")
        print(f"melody_stem_included={manifest['melody_stem_included']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
