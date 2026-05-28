#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = SCRIPT_ROOT.parent / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aif", ".aiff"}
ROLE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("vocals", ("acapella", "acap", "vocals", "vocal")),
    ("instrumental_mix", ("instrumental", "karaoke", "minus", "no_vocals", "novocals")),
    ("drums", ("beat", "drums", "drum", "kick", "hihat")),
    ("bass", ("bass",)),
    ("melody", ("melody", "instru")),
)
RECOMMENDED_BASELINE_ROLES = ("vocals", "instrumental_mix")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path, *, include_sha256: bool = False) -> dict[str, Any]:
    stat = path.stat()
    payload: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "size_bytes": stat.st_size,
        "last_modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
    }
    if include_sha256:
        payload["sha256"] = sha256_file(path)
    return payload


def read_xml_root(path: Path) -> ET.Element:
    if not path.exists():
        raise FileNotFoundError(path)
    return ET.parse(path).getroot()


def child_text(parent: ET.Element | None, child_name: str) -> str | None:
    if parent is None:
        return None
    child = parent.find(child_name)
    if child is None or child.text is None:
        return None
    return child.text


def section_texts(root: ET.Element, section_name: str) -> dict[str, str | None]:
    section = root.find(section_name)
    if section is None:
        return {}
    return {child.tag: child.text for child in section}


def selected_texts(root: ET.Element, section_name: str, names: Sequence[str]) -> dict[str, str | None]:
    section = root.find(section_name)
    return {name: child_text(section, name) for name in names}


def parse_virtualdj_settings(settings_path: Path) -> dict[str, Any]:
    if not settings_path.exists():
        return {
            "exists": False,
            "path": str(settings_path),
        }
    root = read_xml_root(settings_path)
    return {
        "exists": True,
        "file": file_info(settings_path),
        "automation": selected_texts(
            root,
            "automation",
            [
                "autoBPMMatch",
                "autoPitchLock",
                "autoGain",
                "autoKey",
                "autoCue",
                "resetStemsOnLoad",
            ],
        ),
        "stems": selected_texts(
            root,
            "performance",
            [
                "stemsRealtimeSeparation",
                "stemsSavedStems",
                "stemsGPU",
                "stemsSavedFolder",
                "maxStemLength",
                "stemsSpeed",
            ],
        ),
        "stems_controls": selected_texts(
            root,
            "controls",
            [
                "defaultStemsEqMode",
                "stemsBleedMuteVocal",
                "stemsBleedMuteInstru",
                "stemsBleedOnlyVocal",
                "stemsBleedOnlyInstru",
                "stemsSplitLeftRight",
            ],
        ),
        "cdj_export": selected_texts(
            root,
            "browser",
            [
                "cdjExportStemsConfig",
                "cdjExportStems",
                "cdjExportCompatibility",
                "cdjExportAutoSyncCues",
                "cdjExportCuesAsMemoryPoints",
            ],
        ),
        "remote_and_transport": selected_texts(
            root,
            "internet",
            [
                "iRemote",
                "iRemoteDefaultPort",
                "os2l",
                "oscPort",
                "oscPortBack",
                "oscInitSubscribe",
            ],
        ),
        "audio": selected_texts(root, "performance", ["sampleRate", "latency", "ultraLatency"]),
        "record": selected_texts(root, "record", ["recordFormat", "recordQuality", "recordBitDepth"]),
        "options": selected_texts(root, "options", ["lastUsedBuild", "language", "database"]),
    }


def parse_song(song: ET.Element) -> dict[str, Any]:
    pois = []
    for poi in song.findall("Poi"):
        pois.append(dict(poi.attrib))
    return {
        "file_path": song.attrib.get("FilePath"),
        "file_size": song.attrib.get("FileSize"),
        "tags": dict(song.find("Tags").attrib) if song.find("Tags") is not None else {},
        "infos": dict(song.find("Infos").attrib) if song.find("Infos") is not None else {},
        "scan": dict(song.find("Scan").attrib) if song.find("Scan") is not None else {},
        "poi_count": len(pois),
        "pois": pois,
    }


def parse_virtualdj_database(database_path: Path) -> dict[str, Any]:
    if not database_path.exists():
        return {
            "exists": False,
            "path": str(database_path),
        }
    root = read_xml_root(database_path)
    songs = [parse_song(song) for song in root.findall("Song")]
    return {
        "exists": True,
        "file": file_info(database_path),
        "version": root.attrib.get("Version"),
        "song_count": len(songs),
        "songs": songs,
    }


def summarize_history(history_root: Path) -> dict[str, Any]:
    if not history_root.exists():
        return {
            "exists": False,
            "path": str(history_root),
            "files": [],
        }
    files = [file_info(path) for path in sorted(history_root.iterdir()) if path.is_file()]
    return {
        "exists": True,
        "path": str(history_root),
        "file_count": len(files),
        "files": files,
    }


def summarize_cache(cache_root: Path) -> dict[str, Any]:
    if not cache_root.exists():
        return {
            "exists": False,
            "path": str(cache_root),
            "files": [],
        }
    cache_files = []
    for path in sorted(cache_root.iterdir()):
        if not path.is_file():
            continue
        info = file_info(path)
        if path.suffix.lower() == ".db":
            with path.open("rb") as handle:
                header = handle.read(16)
            info["format_hint"] = "sqlite" if header == b"SQLite format 3\x00" else "binary"
        cache_files.append(info)
    return {
        "exists": True,
        "path": str(cache_root),
        "files": cache_files,
    }


def detect_role(path: Path) -> str:
    normalized_name = path.stem.lower().replace("-", "_").replace(" ", "_")
    for role, patterns in ROLE_PATTERNS:
        if any(pattern in normalized_name for pattern in patterns):
            return role
    return "unknown"


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS and not path.name.startswith(".")


def classify_virtualdj_exports(export_root: Path) -> dict[str, Any]:
    if not export_root.exists():
        return {
            "exists": False,
            "export_root": str(export_root),
            "file_count": 0,
            "detected_roles": [],
            "missing_recommended_roles": list(RECOMMENDED_BASELINE_ROLES),
            "selected_reference_minus": None,
            "items": [],
        }
    if not export_root.is_dir():
        raise NotADirectoryError(export_root)

    audio_files = sorted(path for path in export_root.rglob("*") if is_audio_file(path))
    items = []
    for path in audio_files:
        relative_path = path.relative_to(export_root).as_posix()
        role = detect_role(path)
        item = {
            "role": role,
            "relative_path": relative_path,
            "docker_path": f"/virtualdj_export/{relative_path}",
            "file": file_info(path, include_sha256=True),
        }
        items.append(item)

    if len(items) == 1 and items[0]["role"] == "unknown":
        items[0]["role"] = "instrumental_mix"
        items[0]["role_inference"] = "single_audio_file_defaulted_to_instrumental_mix"

    detected_roles = sorted({item["role"] for item in items})
    selected_reference_minus = next((item for item in items if item["role"] == "instrumental_mix"), None)

    return {
        "exists": True,
        "export_root": str(export_root),
        "file_count": len(items),
        "detected_roles": detected_roles,
        "missing_recommended_roles": [role for role in RECOMMENDED_BASELINE_ROLES if role not in detected_roles],
        "selected_reference_minus": selected_reference_minus,
        "items": items,
    }


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def build_manifest(
    *,
    virtualdj_data_root: Path,
    export_root: Path,
    host_probe_json: Path | None = None,
) -> dict[str, Any]:
    settings = parse_virtualdj_settings(virtualdj_data_root / "settings.xml")
    database = parse_virtualdj_database(virtualdj_data_root / "database.xml")
    history = summarize_history(virtualdj_data_root / "History")
    cache = summarize_cache(virtualdj_data_root / "Cache")
    exports = classify_virtualdj_exports(export_root)
    selected_reference = exports["selected_reference_minus"]

    return {
        "schema_version": 1,
        "execution": {
            "tool": "ml_worker_improved/scripts/virtualdj_baseline_manifest.py",
            "generated_at_utc": utc_timestamp(),
            "intended_runtime": "docker ml-worker for parsing mounted VirtualDJ data/export artifacts",
            "threshold_exit_gate": "not_applicable",
        },
        "virtualdj": {
            "data_root": str(virtualdj_data_root),
            "host_probe": load_optional_json(host_probe_json),
            "settings": settings,
            "database": database,
            "history": history,
            "cache": cache,
        },
        "baseline_exports": exports,
        "quality_gate_hint": {
            "can_run_nine_thou_proxy_gate": selected_reference is not None,
            "reference_minus_docker_path": selected_reference["docker_path"] if selected_reference else None,
            "reason": "instrumental_mix_export_found" if selected_reference else "no_instrumental_mix_export_found",
        },
        "interpretation_limits": [
            "VirtualDJ exports are model-generated or user-exported baselines, not verified isolated studio ground truth by themselves.",
            "Use Acapella/Instrumental exports for repeatable A/B and proxy regression, not final SDR/SI-SDR benchmark claims.",
            "True objective quality still requires a verified ground_truth_manifest with isolated stems and SHA-256 integrity checks.",
            "This manifest parser reads VirtualDJ XML/cache/export artifacts; it does not control the proprietary GUI.",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a machine-readable manifest from mounted VirtualDJ settings/database and exported stem artifacts."
    )
    parser.add_argument("--virtualdj-data-root", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--host-probe-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="virtualdj_baseline_manifest.py",
        entrypoint_path="/app/scripts/virtualdj_baseline_manifest.py",
    )
    args = parse_args(argv)
    if not args.virtualdj_data_root.exists():
        raise FileNotFoundError(args.virtualdj_data_root)
    manifest = build_manifest(
        virtualdj_data_root=args.virtualdj_data_root,
        export_root=args.export_dir,
        host_probe_json=args.host_probe_json,
    )
    write_json(args.output_json, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
