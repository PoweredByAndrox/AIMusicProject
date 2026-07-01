from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping


def append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "phase": phase, "status": status, **extra}, sort_keys=True) + "\n")
        handle.flush()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def path_from_mapping(mapping: Mapping[str, Any], key: str) -> Path:
    value = str(mapping.get(key) or "")
    if not value:
        raise ValueError(f"missing path field: {key}")
    return Path(value)


def blocked_route_manifest(
    *,
    schema: str,
    presentation_status: str,
    residual_manifest_path: Path,
    output_dir: Path,
    status: str,
    failed_checks: list[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema": schema,
        "status": status,
        "failed_checks": list(failed_checks),
        "source_residual_manifest": str(residual_manifest_path),
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": presentation_status,
        **dict(extra or {}),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
