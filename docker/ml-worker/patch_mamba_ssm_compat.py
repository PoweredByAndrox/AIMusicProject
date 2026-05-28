from __future__ import annotations

import site
from pathlib import Path


PATCHED_INIT = '''__version__ = "2.3.2.post1"

from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except Exception:
    MambaLMHeadModel = None

__all__ = [
    "Mamba",
    "Mamba2",
    "MambaLMHeadModel",
    "mamba_inner_fn",
    "selective_scan_fn",
]
'''


def find_mamba_init() -> Path:
    for site_packages_path in site.getsitepackages():
        candidate = Path(site_packages_path) / "mamba_ssm" / "__init__.py"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Installed mamba_ssm package was not found")


def main() -> None:
    init_path = find_mamba_init()
    original = init_path.read_text(encoding="utf-8")
    if "from mamba_ssm.modules.mamba3 import Mamba3" not in original:
        print(f"mamba_ssm compat patch skipped: no Mamba3 import in {init_path}")
        return
    init_path.write_text(PATCHED_INIT, encoding="utf-8")
    print(f"mamba_ssm compat patch applied: {init_path}")


if __name__ == "__main__":
    main()
