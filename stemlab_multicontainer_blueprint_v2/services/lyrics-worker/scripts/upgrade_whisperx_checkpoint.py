from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
from lightning.pytorch.utilities.migration.utils import migrate_checkpoint, pl_legacy_patch


def upgrade_checkpoint(path_str: str) -> int:
    path = Path(path_str)
    if not path.exists():
        print(f"checkpoint not found: {path}", file=sys.stderr)
        return 1

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    with pl_legacy_patch():
        checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    migrate_checkpoint(checkpoint)
    torch.save(checkpoint, path)
    print(f"upgraded checkpoint: {path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: upgrade_whisperx_checkpoint.py <checkpoint_path>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(upgrade_checkpoint(sys.argv[1]))
