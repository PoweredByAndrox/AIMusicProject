from pathlib import Path
from typing import Any, Dict, List
import yaml

from app.config import CONFIG_ROOT


def list_presets() -> List[str]:
    preset_dir = CONFIG_ROOT / "presets"
    return sorted([p.stem for p in preset_dir.glob("*.yaml")])


def load_preset(name: str) -> Dict[str, Any]:
    path = CONFIG_ROOT / "presets" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {name}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_model_registry() -> Dict[str, Any]:
    path = CONFIG_ROOT / "model_registry.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
