from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

DEFAULT_REPO_PATH = Path("/app/vendor/music_source_separation_training")
DEFAULT_MODEL_DIR = Path("/app/vendor/bs_mamba2_vocals")
DEFAULT_CONFIG_PATH = DEFAULT_MODEL_DIR / "config_bs_mamba2_vocals.yaml"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "bs_mamba2_vocals.ckpt"


def main() -> None:
    require_docker_runtime(
        tool_name="export_bs_mamba2_vocal_preview.py",
        entrypoint_path="/app/tools/export_bs_mamba2_vocal_preview.py",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("source_audio_path")
    parser.add_argument("output_dir")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=40.0)
    parser.add_argument("--repo-path", default=str(DEFAULT_REPO_PATH))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-overlap", type=int, default=4)
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="01_bs_mamba2_vocals.flac")
    args = parser.parse_args()

    sibling = Path(__file__).with_name("export_dereverb_vocal_preview.py")
    command = [
        sys.executable,
        str(sibling),
        str(args.source_audio_path),
        str(args.output_dir),
        "--start-sec",
        str(args.start_sec),
        "--end-sec",
        str(args.end_sec),
        "--repo-path",
        str(args.repo_path),
        "--config-path",
        str(args.config_path),
        "--model-path",
        str(args.model_path),
        "--model-type",
        "bs_mamba2",
        "--target-output",
        "vocals",
        "--device-id",
        str(args.device_id),
        "--num-overlap",
        str(args.num_overlap),
    ]
    if args.listen_dir:
        command.extend(["--listen-dir", str(args.listen_dir)])
    if args.listen_label:
        command.extend(["--listen-label", str(args.listen_label)])

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
