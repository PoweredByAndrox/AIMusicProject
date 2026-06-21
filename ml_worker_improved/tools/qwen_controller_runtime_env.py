from __future__ import annotations

import os


def configure_pytorch_only_transformers_env() -> dict[str, str]:
    os.environ["USE_TF"] = "0"
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    return {
        "USE_TF": str(os.environ.get("USE_TF") or ""),
        "TRANSFORMERS_NO_TF": str(os.environ.get("TRANSFORMERS_NO_TF") or ""),
    }
