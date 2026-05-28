from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BUILD_MODEL = "hybrid_htdemucs_ft_6s"
SUPPORTED_STEM_ORDER = ("drums", "bass", "other", "vocals", "guitar", "piano")


def _stem_payload(
    stem_key: str,
    stem_relative_path: str,
    *,
    family: str,
) -> dict[str, Any]:
    label = stem_key.replace("_", " ").title()

    return {
        "label": label,
        "description": f"Auto-built {label} stem from the 6-stem container export.",
        "family": family,
        "kind": "stem",
        "default_enabled": True,
        "path": stem_relative_path,
    }


def build_track_playback_stems(
    audio_path: str | Path,
    output_dir: str | Path,
    playback_manifest_path: str | Path,
    *,
    build_id: str,
    latest_pointer_path: str | Path,
    model_name: str = DEFAULT_BUILD_MODEL,
    device: str = "cuda",
) -> dict[str, Any]:
    resolved_audio_path = Path(audio_path)
    resolved_output_dir = Path(output_dir)
    resolved_playback_manifest_path = Path(playback_manifest_path)
    resolved_latest_pointer_path = Path(latest_pointer_path)
    storage_root = Path("/app/storage").resolve()

    if not resolved_audio_path.exists():
        raise FileNotFoundError(f"Track source audio is missing: {resolved_audio_path}")

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_playback_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_latest_pointer_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = resolved_output_dir / "build_summary.json"
    command = [
        sys.executable,
        "/app/tools/export_hq_stems_full.py",
        str(resolved_audio_path),
        str(resolved_output_dir),
        "--model-name",
        str(model_name),
        "--device",
        str(device),
        "--sharpen-bleed",
        "--no-melody-stem",
        "--summary-path",
        str(summary_path),
    ]

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"Stem source build failed: {detail}")

    stem_manifest_path = resolved_output_dir / "stem_manifest.json"
    if not stem_manifest_path.exists():
        raise FileNotFoundError(f"Stem build finished without a manifest: {stem_manifest_path}")

    stem_manifest = json.loads(stem_manifest_path.read_text(encoding="utf-8"))
    stems_payload = stem_manifest.get("stems", {})
    if not isinstance(stems_payload, dict) or stems_payload == {}:
        raise ValueError(f"Stem build manifest has no stems: {stem_manifest_path}")

    playback_stems: dict[str, Any] = {}
    built_stems: list[str] = []
    for stem_key in SUPPORTED_STEM_ORDER:
        raw_stem = stems_payload.get(stem_key)
        if not isinstance(raw_stem, dict):
            continue

        source_path = Path(str(raw_stem.get("path", ""))).resolve()
        if not source_path.exists():
            continue

        try:
            relative_stem_path = source_path.relative_to(storage_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"Built stem path is outside worker storage: {source_path}") from exc

        playback_stems[stem_key] = _stem_payload(
            stem_key,
            relative_stem_path,
            family=str(model_name),
        )
        built_stems.append(stem_key)

    if playback_stems == {}:
        raise RuntimeError(f"Stem build created no supported 6-stem files: {stem_manifest_path}")

    playback_manifest = {
        "default_mode_key": "original",
        "build": {
            "id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "source_audio_path": str(resolved_audio_path),
            "source_stem_manifest": str(stem_manifest_path),
        },
        "stems": playback_stems,
    }
    resolved_playback_manifest_path.write_text(
        json.dumps(playback_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    resolved_latest_pointer_path.write_text(
        json.dumps({
            "build_id": build_id,
            "track_audio_path": str(resolved_audio_path),
            "playback_manifest_path": str(resolved_playback_manifest_path),
            "stem_manifest_path": str(stem_manifest_path),
            "output_dir": str(resolved_output_dir),
            "built_stems": built_stems,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "build_id": build_id,
        "playback_manifest_path": str(resolved_playback_manifest_path),
        "stem_manifest_path": str(stem_manifest_path),
        "output_dir": str(resolved_output_dir),
        "built_stems": built_stems,
    }
