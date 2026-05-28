#!/usr/bin/env python3
"""Build a verified ground-truth bundle from FL Studio split mixer exports.

The importer is intentionally rule-based and track-agnostic. It maps detailed
split-mixer files into canonical stems by semantic file-name tokens, preserves
audio gain, and records provenance so objective gates can audit the bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402


AUDIO_EXTENSIONS = {".wav", ".flac", ".aif", ".aiff"}
FULL_MIX_RECONSTRUCTION_MIN_CORRELATION = 0.995
FULL_MIX_RECONSTRUCTION_MAX_RESIDUAL_RMS_DBFS = -60.0
FULL_MIX_RECONSTRUCTION_MAX_RESIDUAL_PEAK_DBFS = -20.0
INPUT_SCOPE_MIN_TRACK_PREFIX_OVERLAP = 0.8
REPORT_ALIGNMENT_SOURCE_DOCUMENT = "deep-research-report.md"
DEFAULT_REPORT_REQUIRED_QC = (
    "sample_rate_consistency",
    "sample_accurate_alignment",
    "loudness_and_peak_summary",
    "tail_integrity_summary",
    "full_mix_reconstruction_contract",
)
CANONICAL_STEMS = (
    "vocals",
    "drums",
    "bass",
    "piano",
    "guitar",
    "strings_orchestra",
    "melody",
    "fx",
    "other",
)
DRUMS_PLUS_PERCUSSIVE_FX_AUX_STEM = "drums_plus_percussive_fx_aux"
DERIVED_STEMS = ("instrumental", DRUMS_PLUS_PERCUSSIVE_FX_AUX_STEM)

VOCAL_TOKENS = ("vocal", "vocals", "vox", "acapella", "acappella", "leadvox", "backingvox")
FX_TOKENS = (
    "fx",
    "riser",
    "sweep",
    "reverse",
    "impact",
    "hit",
    "noise",
    "echo",
    "delay",
    "uplifter",
    "downlifter",
)
DRUM_TOKENS = (
    "kick",
    "snare",
    "clap",
    "hat",
    "hats",
    "hihat",
    "cymbal",
    "crash cymbal",
    "shaker",
    "tom",
    "toms",
    "perc",
    "percussion",
    "drum",
    "drums",
)
BASS_TOKENS = ("bass", "sub")
PIANO_TOKENS = ("piano", "keys")
GUITAR_TOKENS = ("guitar",)
STRING_TOKENS = ("string", "strings", "orchestra", "orchestral", "violin", "viola", "cello", "brass")
MELODY_TOKENS = (
    "lead",
    "synth",
    "chord",
    "chords",
    "pluck",
    "pad",
    "saw",
    "sine",
    "melody",
    "wobble",
    "arp",
    "arpeggio",
)


@dataclass(frozen=True)
class AudioFile:
    path: Path
    sample_rate: int
    audio: np.ndarray


def _normalize_name(value: str) -> str:
    without_extension = Path(value).stem.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", without_extension)
    return re.sub(r"\s+", " ", normalized).strip()


def _normalized_tokens(value: str) -> set[str]:
    return {token for token in _normalize_name(value).split() if token}


def _extract_stem_label(file_name: str) -> str:
    """Return the FL mixer-track label without the song title prefix when present."""

    stem = Path(file_name).stem
    if " - " in stem:
        return stem.rsplit(" - ", 1)[-1]
    return stem


def _extract_track_prefix(file_name: str) -> str | None:
    stem = Path(file_name).stem
    if " - " not in stem:
        return None
    prefix = stem.rsplit(" - ", 1)[0].strip()
    return prefix or None


def _has_token(normalized_name: str, tokens: Sequence[str]) -> bool:
    padded = f" {normalized_name} "
    return any(f" {token} " in padded for token in tokens)


def classify_flstudio_stem_name(file_name: str) -> str:
    """Map an FL split-mixer file name to a canonical stem without track IDs."""

    normalized = _normalize_name(_extract_stem_label(file_name))
    if _has_token(normalized, VOCAL_TOKENS):
        return "vocals"

    # Riser/sweep/impact files can contain clap/crash words but are mix effects,
    # while actual kick/snare/hat files remain drums.
    if _has_token(normalized, FX_TOKENS) and not (" kick " in f" {normalized} " or " cymbal " in f" {normalized} "):
        return "fx"
    if _has_token(normalized, DRUM_TOKENS):
        return "drums"
    if _has_token(normalized, BASS_TOKENS):
        return "bass"
    if _has_token(normalized, PIANO_TOKENS):
        return "piano"
    if _has_token(normalized, GUITAR_TOKENS):
        return "guitar"
    if _has_token(normalized, STRING_TOKENS):
        return "strings_orchestra"
    if _has_token(normalized, FX_TOKENS):
        return "fx"
    if _has_token(normalized, MELODY_TOKENS):
        return "melody"
    return "other"


def is_percussive_fx_aux_name(file_name: str) -> bool:
    """Return true for FX-labeled mixer tracks that are also percussion-like."""

    normalized = _normalize_name(_extract_stem_label(file_name))
    return _has_token(normalized, FX_TOKENS) and _has_token(normalized, DRUM_TOKENS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_audio(path: Path) -> AudioFile:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return AudioFile(path=path, sample_rate=int(sample_rate), audio=np.asarray(audio, dtype=np.float32))


def _align_audio(audio: np.ndarray, target_frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    original_frames = int(audio.shape[0])
    if original_frames == target_frames:
        return audio, {"action": "none", "original_frames": original_frames, "target_frames": target_frames}
    if original_frames > target_frames:
        return audio[:target_frames], {
            "action": "truncated",
            "original_frames": original_frames,
            "target_frames": target_frames,
            "removed_frames": original_frames - target_frames,
        }
    padding = np.zeros((target_frames - original_frames, audio.shape[1]), dtype=np.float32)
    return np.concatenate([audio, padding], axis=0), {
        "action": "padded",
        "original_frames": original_frames,
        "target_frames": target_frames,
        "added_frames": target_frames - original_frames,
    }


def _rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64))))) if audio.size else 0.0
    if rms <= 1e-12:
        return -240.0
    return round(20.0 * math.log10(rms), 6)


def _peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-12:
        return -240.0
    return round(20.0 * math.log10(peak), 6)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.astype(np.float64).reshape(-1)
    b_flat = b.astype(np.float64).reshape(-1)
    a_centered = a_flat - float(np.mean(a_flat))
    b_centered = b_flat - float(np.mean(b_flat))
    denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denominator <= 1e-12:
        return 0.0
    return round(float(np.dot(a_centered, b_centered) / denominator), 9)


def _audio_summary(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    return {
        "duration_sec": round(audio.shape[0] / float(sample_rate), 6),
        "frames": int(audio.shape[0]),
        "channels": int(audio.shape[1]) if audio.ndim == 2 else 1,
        "rms_dbfs": _rms_dbfs(audio),
        "peak_dbfs": _peak_dbfs(audio),
    }


def _full_mix_reconstruction_contract(reference_sum_check: dict[str, float]) -> dict[str, Any]:
    correlation = float(reference_sum_check["source_vs_sum_correlation"])
    residual_rms_dbfs = float(reference_sum_check["sum_minus_source_rms_dbfs"])
    residual_peak_dbfs = float(reference_sum_check["sum_minus_source_peak_dbfs"])

    failed_check_names: list[str] = []
    if correlation < FULL_MIX_RECONSTRUCTION_MIN_CORRELATION:
        failed_check_names.append("correlation_below_threshold")
    if residual_rms_dbfs > FULL_MIX_RECONSTRUCTION_MAX_RESIDUAL_RMS_DBFS:
        failed_check_names.append("residual_rms_above_threshold")
    if residual_peak_dbfs > FULL_MIX_RECONSTRUCTION_MAX_RESIDUAL_PEAK_DBFS:
        failed_check_names.append("residual_peak_above_threshold")

    can_claim_source_mix_1_to_1 = not failed_check_names
    problem_summary = (
        []
        if can_claim_source_mix_1_to_1
        else ["Source full mix is not verified as the exact sum of imported stems."]
    )
    return {
        "status": "accepted" if can_claim_source_mix_1_to_1 else "rejected",
        "mode": "source_mix_equals_imported_stem_sum",
        "can_claim_source_mix_1_to_1": can_claim_source_mix_1_to_1,
        "failed_check_names": failed_check_names,
        "thresholds": {
            "min_source_vs_sum_correlation": FULL_MIX_RECONSTRUCTION_MIN_CORRELATION,
            "max_sum_minus_source_rms_dbfs": FULL_MIX_RECONSTRUCTION_MAX_RESIDUAL_RMS_DBFS,
            "max_sum_minus_source_peak_dbfs": FULL_MIX_RECONSTRUCTION_MAX_RESIDUAL_PEAK_DBFS,
        },
        "problem_summary": problem_summary,
    }


def _write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_24")


def _relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except ValueError:
        return str(path)


def _normalized_metadata_list(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def _source_identity_values(source_mix_path: Path, track_name: str | None) -> list[str]:
    values: list[str] = []
    for value in (track_name, source_mix_path.stem):
        if not value:
            continue
        normalized = _normalize_name(str(value))
        if normalized and normalized not in values:
            values.append(normalized)
    return values


def _source_identity_tokens(source_mix_path: Path, track_name: str | None) -> set[str]:
    tokens: set[str] = set()
    for value in _source_identity_values(source_mix_path, track_name):
        tokens.update(_normalized_tokens(value))
    return tokens


def _track_prefix_overlap(prefix_tokens: set[str], source_tokens: set[str]) -> float:
    if not prefix_tokens or not source_tokens:
        return 0.0
    return round(len(prefix_tokens & source_tokens) / float(len(prefix_tokens)), 6)


def _build_input_scope_contract(
    *,
    source_mix_path: Path,
    stem_paths: Sequence[Path],
    track_name: str | None,
    allow_mixed_stem_dir: bool,
) -> dict[str, Any]:
    source_values = _source_identity_values(source_mix_path, track_name)
    source_tokens = _source_identity_tokens(source_mix_path, track_name)
    suspicious_files: list[dict[str, Any]] = []

    for stem_path in stem_paths:
        prefix = _extract_track_prefix(stem_path.name)
        if prefix is None:
            assigned_stem = classify_flstudio_stem_name(stem_path.name)
            if assigned_stem == "other":
                suspicious_files.append(
                    {
                        "path": str(stem_path),
                        "name": stem_path.name,
                        "track_prefix": None,
                        "normalized_track_prefix": None,
                        "overlap_ratio": 0.0,
                        "reason": "unscoped_unknown_audio",
                    }
                )
            continue
        prefix_tokens = _normalized_tokens(prefix)
        overlap = _track_prefix_overlap(prefix_tokens, source_tokens)
        if overlap >= INPUT_SCOPE_MIN_TRACK_PREFIX_OVERLAP:
            continue
        suspicious_files.append(
            {
                "path": str(stem_path),
                "name": stem_path.name,
                "track_prefix": prefix,
                "normalized_track_prefix": _normalize_name(prefix),
                "overlap_ratio": overlap,
                "reason": "foreign_track_prefix",
            }
        )

    status = "accepted"
    if suspicious_files:
        status = "accepted_with_warnings" if allow_mixed_stem_dir else "rejected"

    return {
        "status": status,
        "mode": "strict_track_prefix_guard",
        "allow_mixed_stem_dir": bool(allow_mixed_stem_dir),
        "min_track_prefix_overlap": INPUT_SCOPE_MIN_TRACK_PREFIX_OVERLAP,
        "source_identity_values": source_values,
        "source_identity_tokens": sorted(source_tokens),
        "source_file_candidate_count": len(stem_paths),
        "suspicious_foreign_track_files": suspicious_files,
        "notes": [
            "Default strict mode rejects files whose FL export track prefix does not match the source mix identity.",
            "Use allow_mixed_stem_dir only for legacy or forensic mixed-directory imports; such bundles must not support clean exact-pair claims without further audit.",
        ],
    }


def _raise_for_rejected_input_scope(input_scope_contract: dict[str, Any]) -> None:
    if input_scope_contract["status"] != "rejected":
        return
    suspicious_names = ", ".join(
        item["name"] for item in input_scope_contract["suspicious_foreign_track_files"]
    )
    suspicious_reasons = ", ".join(
        sorted({item["reason"].replace("_", " ") for item in input_scope_contract["suspicious_foreign_track_files"]})
    )
    raise ValueError(
        "Stem directory contains files outside the current bundle input scope "
        f"({suspicious_reasons}): "
        f"{suspicious_names}. Move unrelated audio out of stem_dir, provide a clean export directory, "
        "or pass --allow-mixed-stem-dir for a diagnostic mixed-directory import."
    )


def _iter_source_stem_files(source_mix_path: Path, stem_dir: Path) -> Iterable[Path]:
    source_resolved = source_mix_path.resolve()
    for path in sorted(stem_dir.iterdir(), key=lambda candidate: candidate.name.lower()):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if path.resolve() == source_resolved:
            continue
        yield path


def build_flstudio_ground_truth_bundle(
    *,
    source_mix_path: Path,
    stem_dir: Path,
    output_dir: Path,
    track_name: str | None = None,
    bundle_kind: str = "clean",
    master_chain_status: str = "unknown",
    rights_status: str = "unknown",
    license_notes: str | None = None,
    validation_pack_tags: Sequence[str] | None = None,
    hard_case_tags: Sequence[str] | None = None,
    allow_mixed_stem_dir: bool = False,
) -> dict[str, Any]:
    source_mix_path = source_mix_path.resolve()
    stem_dir = stem_dir.resolve()
    output_dir = output_dir.resolve()

    if not source_mix_path.exists():
        raise FileNotFoundError(f"Source mix is missing: {source_mix_path}")
    if not stem_dir.exists():
        raise FileNotFoundError(f"Stem directory is missing: {stem_dir}")

    source_mix = _read_audio(source_mix_path)
    target_frames = int(source_mix.audio.shape[0])
    sample_rate = int(source_mix.sample_rate)
    stem_paths = list(_iter_source_stem_files(source_mix_path, stem_dir))
    if not stem_paths:
        raise RuntimeError(f"No source stem files found in {stem_dir}")
    input_scope_contract = _build_input_scope_contract(
        source_mix_path=source_mix_path,
        stem_paths=stem_paths,
        track_name=track_name,
        allow_mixed_stem_dir=allow_mixed_stem_dir,
    )
    _raise_for_rejected_input_scope(input_scope_contract)

    output_dir.mkdir(parents=True, exist_ok=True)
    stems_dir = output_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    grouped_audio = {
        stem_name: np.zeros_like(source_mix.audio, dtype=np.float32)
        for stem_name in CANONICAL_STEMS
    }
    percussive_fx_aux_audio = np.zeros_like(source_mix.audio, dtype=np.float32)
    percussive_fx_aux_source_files: list[str] = []
    source_files: list[dict[str, Any]] = []
    source_sum = np.zeros_like(source_mix.audio, dtype=np.float32)

    for stem_path in stem_paths:
        stem_audio_file = _read_audio(stem_path)
        if stem_audio_file.sample_rate != sample_rate:
            raise ValueError(
                f"Sample-rate mismatch: stem={stem_path} stem_sample_rate={stem_audio_file.sample_rate} "
                f"source_sample_rate={sample_rate}"
            )
        aligned_audio, alignment = _align_audio(stem_audio_file.audio, target_frames)
        assigned_stem = classify_flstudio_stem_name(stem_path.name)
        grouped_audio[assigned_stem] = grouped_audio[assigned_stem] + aligned_audio
        target_memberships: list[str] = []
        if assigned_stem == "fx" and is_percussive_fx_aux_name(stem_path.name):
            percussive_fx_aux_audio = percussive_fx_aux_audio + aligned_audio
            percussive_fx_aux_source_files.append(stem_path.name)
            target_memberships.append(DRUMS_PLUS_PERCUSSIVE_FX_AUX_STEM)
        source_sum = source_sum + aligned_audio
        source_files.append(
            {
                "path": str(stem_path),
                "name": stem_path.name,
                "assigned_stem": assigned_stem,
                "target_memberships": target_memberships,
                "sha256": sha256_file(stem_path),
                "alignment": alignment,
                **_audio_summary(aligned_audio, sample_rate),
            }
        )

    grouped_audio["instrumental"] = sum(
        grouped_audio[stem_name]
        for stem_name in CANONICAL_STEMS
        if stem_name != "vocals"
    ).astype(np.float32)
    if percussive_fx_aux_source_files:
        grouped_audio[DRUMS_PLUS_PERCUSSIVE_FX_AUX_STEM] = (
            grouped_audio["drums"] + percussive_fx_aux_audio
        ).astype(np.float32)

    source_copy_path = output_dir / "source_mix.flac"
    reference_sum_mix_path = output_dir / "reference_sum_mix.flac"
    _write_flac(source_copy_path, source_mix.audio, sample_rate)
    _write_flac(reference_sum_mix_path, source_sum, sample_rate)

    stem_entries: dict[str, Any] = {}
    derived_stem_names = ("instrumental",)
    if percussive_fx_aux_source_files:
        derived_stem_names = (*derived_stem_names, DRUMS_PLUS_PERCUSSIVE_FX_AUX_STEM)

    for stem_name in (*CANONICAL_STEMS, *derived_stem_names):
        stem_audio = grouped_audio[stem_name]
        if stem_name not in DERIVED_STEMS and float(np.max(np.abs(stem_audio))) <= 1e-8:
            continue
        stem_path = stems_dir / f"{stem_name}.flac"
        _write_flac(stem_path, stem_audio, sample_rate)
        stem_entries[stem_name] = {
            "path": _relative_path(stem_path, output_dir),
            "sha256": sha256_file(stem_path),
            "derived": stem_name in DERIVED_STEMS,
            "reference_role": (
                "composite_reference_target"
                if stem_name == DRUMS_PLUS_PERCUSSIVE_FX_AUX_STEM
                else "individual_stem_reference_target"
            ),
            **_audio_summary(stem_audio, sample_rate),
        }

    residual = source_sum - source_mix.audio
    reference_sum_check = {
        "source_vs_sum_correlation": _correlation(source_mix.audio, source_sum),
        "sum_minus_source_rms_dbfs": _rms_dbfs(residual),
        "sum_minus_source_peak_dbfs": _peak_dbfs(residual),
        "source_sum_rms_dbfs": _rms_dbfs(source_sum),
        "source_mix_rms_dbfs": _rms_dbfs(source_mix.audio),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "track_name": track_name or source_mix_path.stem,
        "sample_rate": sample_rate,
        "source_file_count": len(source_files),
        "input_scope_contract": input_scope_contract,
        "ground_truth_contract": {
            "verified": True,
            "mode": "verified_isolated_stems",
            "verification_method": "flstudio_split_mixer_tracks_import",
            "source": "FL Studio split mixer tracks plus full mix supplied by user",
            "notes": [
                "Reference stems are grouped from detailed split mixer files by track-agnostic semantic filename rules.",
                "Imported stems are individual reference targets for per-stem separation, not a recipe for rebuilding the authoritative full mix.",
                "Derived instrumental stem is a convenience target and must not be evaluated together with its child stems.",
            ],
        },
        "mix": {
            "role": "authoritative_full_mix_input",
            "path": _relative_path(source_copy_path, output_dir),
            "sha256": sha256_file(source_copy_path),
            **_audio_summary(source_mix.audio, sample_rate),
        },
        "reference_sum_mix": {
            "role": "diagnostic_imported_stem_sum",
            "not_authoritative_mix": True,
            "path": _relative_path(reference_sum_mix_path, output_dir),
            "sha256": sha256_file(reference_sum_mix_path),
            **_audio_summary(source_sum, sample_rate),
        },
        "stems": stem_entries,
        "target_definitions": {
            DRUMS_PLUS_PERCUSSIVE_FX_AUX_STEM: {
                "derived": True,
                "reference_role": "composite_reference_target",
                "description": "Strict drums plus FX tracks whose mixer labels are both percussive and FX-like.",
                "child_stems": ["drums", "fx"],
                "aux_source_files": sorted(percussive_fx_aux_source_files),
                "selection_rule": "assigned_stem == fx and filename contains at least one DRUM token and one FX token",
            }
        } if percussive_fx_aux_source_files else {},
        "source_files": source_files,
        "reference_sum_check": reference_sum_check,
        "full_mix_reconstruction_contract": _full_mix_reconstruction_contract(reference_sum_check),
        "report_alignment_contract": {
            "source_document": REPORT_ALIGNMENT_SOURCE_DOCUMENT,
            "implemented_report_sections": [
                "data_domain_shift_engineering_issues",
                "experimental_plan_and_validation",
                "local_pipeline_and_fl_studio_integration",
                "research_limitations_and_legal_aspects",
            ],
            "bundle_kind": bundle_kind,
            "master_chain_status": master_chain_status,
            "rights_status": rights_status,
            "license_notes": license_notes,
            "validation_pack_tags": _normalized_metadata_list(validation_pack_tags),
            "hard_case_tags": _normalized_metadata_list(hard_case_tags),
            "required_qc": list(DEFAULT_REPORT_REQUIRED_QC),
            "notes": [
                "Clean, mastered, and hard-case bundles must be identified explicitly before benchmark aggregation.",
                "Rights status is provenance metadata only; it does not grant redistribution rights for source audio or derived stems.",
                "Human AB/MUSHRA evaluation remains a separate validation gate and is not implied by this bundle import.",
            ],
        },
    }

    manifest_path = output_dir / "ground_truth_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a verified ground-truth bundle from FL Studio split stems.")
    parser.add_argument("source_mix_path", type=Path)
    parser.add_argument("stem_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--track-name")
    parser.add_argument(
        "--bundle-kind",
        choices=("clean", "mastered", "hard_case", "diagnostic"),
        default="clean",
        help="Report-aligned validation bucket for this FL Studio export bundle.",
    )
    parser.add_argument(
        "--master-chain-status",
        choices=("unknown", "disabled_clean_export", "mastered_export", "augmentation_stage"),
        default="unknown",
        help="Whether this bundle is clean premaster material, mastered material, or augmentation-stage material.",
    )
    parser.add_argument(
        "--rights-status",
        choices=("unknown", "user_owned", "licensed", "research_only", "noncommercial", "blocked"),
        default="unknown",
        help="Rights/provenance status for source audio and stems; recorded for audit, not used as permission grant.",
    )
    parser.add_argument("--license-notes")
    parser.add_argument("--validation-pack-tag", action="append", default=[])
    parser.add_argument("--hard-case-tag", action="append", default=[])
    parser.add_argument(
        "--allow-mixed-stem-dir",
        action="store_true",
        help=(
            "Allow files whose FL export track prefix does not match the source mix identity. "
            "Use only for legacy/forensic mixed-directory imports; strict mode rejects them by default."
        ),
    )
    parser.add_argument(
        "--require-full-mix-reconstruction",
        action="store_true",
        help="Return non-zero unless the imported stems reconstruct the supplied source mix.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_flstudio_ground_truth_bundle.py",
        entrypoint_path="/app/tools/build_flstudio_ground_truth_bundle.py",
    )
    args = parse_args(argv)
    manifest = build_flstudio_ground_truth_bundle(
        source_mix_path=args.source_mix_path,
        stem_dir=args.stem_dir,
        output_dir=args.output_dir,
        track_name=args.track_name,
        bundle_kind=args.bundle_kind,
        master_chain_status=args.master_chain_status,
        rights_status=args.rights_status,
        license_notes=args.license_notes,
        validation_pack_tags=args.validation_pack_tag,
        hard_case_tags=args.hard_case_tag,
        allow_mixed_stem_dir=args.allow_mixed_stem_dir,
    )
    print(
        json.dumps(
            {
                "manifest_path": manifest["manifest_path"],
                "track_name": manifest["track_name"],
                "source_file_count": manifest["source_file_count"],
                "stems": sorted(manifest["stems"].keys()),
                "reference_sum_check": manifest["reference_sum_check"],
                "full_mix_reconstruction_contract": manifest["full_mix_reconstruction_contract"],
            },
            indent=2,
        )
    )
    if args.require_full_mix_reconstruction and manifest["full_mix_reconstruction_contract"]["status"] != "accepted":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
