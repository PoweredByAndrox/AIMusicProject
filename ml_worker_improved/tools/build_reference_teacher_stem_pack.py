from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import soundfile as sf


ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = ROOT_DIR / "scripts"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
import run_ground_truth_quality_gate as quality_gate  # noqa: E402


MODEL_ID = "reference_teacher_stem_pack"
DERIVED_STEMS = frozenset({"instrumental", "drums_plus_percussive_fx_aux"})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_file_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    info = sf.info(str(path))
    summary.update(
        {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "frames": int(info.frames),
            "duration_sec": round(float(info.duration), 6),
            "format": str(info.format),
            "subtype": str(info.subtype),
        }
    )
    return summary


def resolve_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def resolve_ground_truth_mix_path(ground_truth_manifest: Path, manifest: dict[str, Any]) -> Path:
    mix = manifest.get("mix")
    if not isinstance(mix, dict) or not mix.get("path"):
        raise RuntimeError(f"Ground-truth manifest has no mix.path: {ground_truth_manifest}")
    mix_path = resolve_manifest_path(ground_truth_manifest, str(mix["path"]))
    if not mix_path.exists():
        raise FileNotFoundError(mix_path)
    return mix_path


def stem_is_derived(stem_name: str, stem_entry: Any) -> bool:
    if isinstance(stem_entry, dict) and isinstance(stem_entry.get("derived"), bool):
        return bool(stem_entry["derived"])
    return stem_name in DERIVED_STEMS


def select_teacher_stems(
    *,
    manifest: dict[str, Any],
    requested_stems: Sequence[str],
    include_derived: bool,
) -> list[str]:
    stems = manifest.get("stems")
    if not isinstance(stems, dict) or not stems:
        raise RuntimeError("Ground-truth manifest has no stems object")

    if requested_stems:
        selected: list[str] = []
        for stem_name in requested_stems:
            if stem_name not in stems:
                raise RuntimeError(f"Requested stem is missing from verified ground truth: {stem_name}")
            if stem_is_derived(stem_name, stems[stem_name]) and not include_derived:
                raise RuntimeError(f"Requested stem is derived; pass --include-derived: {stem_name}")
            selected.append(stem_name)
        return sorted(dict.fromkeys(selected))

    selected = []
    for stem_name, stem_entry in stems.items():
        if stem_is_derived(str(stem_name), stem_entry) and not include_derived:
            continue
        selected.append(str(stem_name))
    return sorted(selected)


def copy_audio_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def build_listen_pack(
    *,
    source_mix_path: Path,
    manifest_path: Path,
    copied_stems: dict[str, Path],
    listen_dir: Path,
) -> dict[str, Any]:
    listen_dir.mkdir(parents=True, exist_ok=True)
    listen_source_path = listen_dir / "00_source_mix.flac"
    copy_audio_file(source_mix_path, listen_source_path)

    candidate_map: dict[str, Any] = {
        "schema_version": 1,
        "contract": {
            "listen_pack_type": "reference_teacher_full_stem_pack",
            "target_reference_assisted": True,
            "delivery_allowed_without_reference": False,
            "not_a_no_reference_production_route": True,
        },
        "source": {
            "label": "00_source_mix.flac",
            "path": str(listen_source_path),
        },
        "candidate_manifest": str(manifest_path),
        "stems": {},
    }

    readme_lines = [
        "Listen only here.",
        "00 = verified source mix supplied with the FL Studio ground-truth bundle.",
        "Following files are copied verified teacher/reference stems, not no-reference production output.",
    ]
    for index, stem_name in enumerate(sorted(copied_stems), start=1):
        listen_label = f"{index:02d}_{stem_name}_reference_teacher.flac"
        listen_path = listen_dir / listen_label
        copy_audio_file(copied_stems[stem_name], listen_path)
        candidate_map["stems"][stem_name] = {
            "label": listen_label,
            "path": str(listen_path),
            "role": "verified_reference_teacher_target",
        }
        readme_lines.append(f"{listen_label} = verified reference teacher stem `{stem_name}`.")

    (listen_dir / "00_README.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    write_json(listen_dir / "candidate_map.json", candidate_map)
    return candidate_map


def build_reference_teacher_stem_pack(
    *,
    ground_truth_manifest: Path,
    output_dir: Path,
    requested_stems: Sequence[str],
    include_derived: bool,
    listen_dir: Path | None,
) -> dict[str, Any]:
    ground_truth_manifest = ground_truth_manifest.resolve()
    output_dir = output_dir.resolve()
    bundle = quality_gate.resolve_stem_bundle(ground_truth_manifest)
    ground_truth_contract = quality_gate.validate_verified_ground_truth(bundle)
    source_mix_path = resolve_ground_truth_mix_path(ground_truth_manifest, bundle.manifest)
    selected_stems = select_teacher_stems(
        manifest=bundle.manifest,
        requested_stems=requested_stems,
        include_derived=include_derived,
    )
    if not selected_stems:
        raise RuntimeError("No verified teacher stems selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_copy_path = output_dir / "source_mix.flac"
    copy_audio_file(source_mix_path, source_copy_path)

    stem_manifest_entries: dict[str, Any] = {}
    copied_stems: dict[str, Path] = {}
    raw_stems = bundle.manifest["stems"]
    for stem_name in selected_stems:
        source_stem_path = bundle.stems[stem_name]
        target_stem_path = output_dir / "stems" / f"{stem_name}.flac"
        copy_audio_file(source_stem_path, target_stem_path)
        derived = stem_is_derived(stem_name, raw_stems.get(stem_name))
        copied_stems[stem_name] = target_stem_path
        stem_manifest_entries[stem_name] = {
            "path": str(target_stem_path),
            "role": "verified_reference_teacher_target",
            "derived": derived,
            **audio_file_summary(target_stem_path),
        }

    manifest_path = output_dir / "stem_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_audio_path": str(source_copy_path),
        "model_id": MODEL_ID,
        "model_name": MODEL_ID,
        "model": {
            "model_id": MODEL_ID,
            "family": "reference_assisted_teacher",
        },
        "contract": {
            "full_track": True,
            "preview": False,
            "benchmark_compatible": True,
            "fallback_used": False,
            "hidden_inference": False,
            "registered_model_candidate": False,
            "target_reference_assisted": True,
            "delivery_allowed_without_reference": False,
            "per_track_hardcode_allowed": False,
            "reference_teacher_pack": True,
        },
        "reference_assistance": {
            "ground_truth_manifest": str(ground_truth_manifest),
            "ground_truth_contract": ground_truth_contract,
            "source": "verified_isolated_stems",
            "selected_stems": selected_stems,
            "include_derived": bool(include_derived),
            "not_a_no_reference_production_route": True,
        },
        "source_mix": {
            "path": str(source_copy_path),
            **audio_file_summary(source_copy_path),
        },
        "stems": stem_manifest_entries,
    }
    write_json(manifest_path, manifest)
    if listen_dir is not None:
        manifest["listen"] = build_listen_pack(
            source_mix_path=source_copy_path,
            manifest_path=manifest_path,
            copied_stems=copied_stems,
            listen_dir=listen_dir.resolve(),
        )
        write_json(manifest_path, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy verified FL Studio ground-truth stems into a benchmark/listen teacher pack."
    )
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", action="append", default=[])
    parser.add_argument("--include-derived", action="store_true")
    parser.add_argument("--listen-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_reference_teacher_stem_pack.py",
        entrypoint_path="/app/tools/build_reference_teacher_stem_pack.py",
    )
    args = build_arg_parser().parse_args(argv)
    manifest = build_reference_teacher_stem_pack(
        ground_truth_manifest=args.ground_truth_manifest,
        output_dir=args.output_dir,
        requested_stems=args.stem,
        include_derived=bool(args.include_derived),
        listen_dir=args.listen_dir,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / "stem_manifest.json"),
                "model_id": manifest["model_id"],
                "stems": sorted(manifest["stems"]),
                "listen_dir": str(args.listen_dir) if args.listen_dir else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
