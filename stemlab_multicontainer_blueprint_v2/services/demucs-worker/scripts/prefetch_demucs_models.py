from __future__ import annotations

import gc
import os

import demucs.pretrained


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_models() -> list[str]:
    raw = os.getenv("DEMUCS_PREFETCH_MODELS", "htdemucs_ft,htdemucs_6s")
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item]


def main() -> int:
    if not env_bool("DEMUCS_PREFETCH_ON_START", True):
        print("demucs prefetch disabled")
        return 0

    models = parse_models()
    if not models:
        print("demucs prefetch skipped: no models configured")
        return 0

    for model_name in models:
        print(f"prefetch demucs model: {model_name}")
        model = demucs.pretrained.get_model(model_name)
        del model
        gc.collect()

    print("demucs prefetch completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
