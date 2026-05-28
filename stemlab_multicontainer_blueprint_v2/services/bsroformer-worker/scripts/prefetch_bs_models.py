from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch
import yaml


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_model_slugs() -> list[str]:
    raw = os.getenv(
        "BSROFORMER_PREFETCH_MODELS",
        "roformer-model-bs-roformer-sw-by-jarredou,roformer-model-bs-roformer-de-reverb",
    )
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item]


def resolve_paths(models_root: Path, model_slug: str) -> tuple[Path, Path]:
    model_dir = models_root / model_slug
    yaml_candidates = list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.yml"))
    ckpt_candidates = list(model_dir.glob("*.ckpt"))
    if not yaml_candidates or not ckpt_candidates:
        raise FileNotFoundError(f"missing yaml/ckpt for {model_slug} in {model_dir}")
    return yaml_candidates[0], ckpt_candidates[0]


def prefetch_signature(models_root: Path, model_slugs: list[str]) -> list[dict]:
    signature: list[dict] = []
    for model_slug in model_slugs:
        config_path, checkpoint_path = resolve_paths(models_root, model_slug)
        signature.append(
            {
                "model": model_slug,
                "config": {
                    "name": config_path.name,
                    "size": config_path.stat().st_size,
                    "mtime_ns": config_path.stat().st_mtime_ns,
                },
                "checkpoint": {
                    "name": checkpoint_path.name,
                    "size": checkpoint_path.stat().st_size,
                    "mtime_ns": checkpoint_path.stat().st_mtime_ns,
                },
            }
        )
    return signature


def main() -> int:
    if not env_bool("BSROFORMER_PREFETCH_ON_START", True):
        print("bsroformer prefetch disabled")
        return 0

    models_root = Path(os.getenv("MODELS_ROOT", "/app/models"))
    cache_root = Path(os.getenv("BSROFORMER_CACHE_ROOT", "/app/models/bsroformer-worker-cache"))
    cache_root.mkdir(parents=True, exist_ok=True)
    model_slugs = parse_model_slugs()
    if not model_slugs:
        print("bsroformer prefetch skipped: no models configured")
        return 0

    for model_slug in model_slugs:
        config_path, checkpoint_path = resolve_paths(models_root, model_slug)
        print(f"prefetch bsroformer model: {model_slug}")
        with config_path.open("r", encoding="utf-8") as handle:
            yaml.unsafe_load(handle)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"unexpected checkpoint payload for {model_slug}")
        del checkpoint
        gc.collect()

    marker = cache_root / "prefetch_state.json"
    marker.write_text(json.dumps(prefetch_signature(models_root, model_slugs), ensure_ascii=False, indent=2), encoding="utf-8")
    print("bsroformer prefetch completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
